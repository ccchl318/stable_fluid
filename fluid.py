import taichi as ti

import numpy as np

ti.init(arch=ti.cpu)

res = 512
dt = 0.03
p_jacoobi_iters = 500

maxfps = 60

force_radius = res / 20.0
f_strength = 10000.0

_velocities_x = ti.field(float, shape=(res + 1, res))
_velocities_x_new = ti.field(float, shape=(res + 1, res))

_velocities_y = ti.field(float, shape=(res, res + 1))
_velocities_y_new = ti.field(float, shape=(res, res + 1))

_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))

velocity_divs = ti.field(float, shape=(res, res))

_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))


@ti.kernel
def init():
    for i in range(res):
        for j in range(100):
            _dye_buffer[i, j + 30][0] += 1
            _dye_buffer[i, j + 30][1] += 0.5
            _dye_buffer[i, j + 30][2] += 0.5


# 双缓冲区，定义cur和nxt，方便数据交换
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_x_pair = TexPair(_velocities_x, _velocities_x_new)
velocities_y_pair = TexPair(_velocities_y, _velocities_y_new)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def lerp(left, right, frac):
    return (1 - frac) * left + frac * right


@ti.func
def bilerp(x00, x01, x10, x11, x, y):
    x0 = lerp(x00, x10, x)
    x1 = lerp(x01, x11, x)
    return lerp(x0, x1, y)


@ti.func
def sample_p(field, coord):
    x, y = coord
    s, t = x - 0.5, y - 0.5
    ix, iy = ti.floor(s), ti.floor(t)
    fx, fy = s - ix, t - iy

    # 边界插值使用镜像
    index00 = ti.max(0, ti.min(res - 1, ti.Vector([int(ix), int(iy)])))
    index01 = ti.max(0, ti.min(res - 1, ti.Vector([int(ix), int(iy + 1)])))
    index10 = ti.max(0, ti.min(res - 1, ti.Vector([int(ix + 1), int(iy)])))
    index11 = ti.max(0, ti.min(res - 1, ti.Vector([int(ix + 1), int(iy + 1)])))

    return bilerp(field[index00], field[index01], field[index10], field[index11], fx, fy)


@ti.func
def sample_vx(vx_field, coord):
    x, y = coord
    s, t = x, y - 0.5
    ix, iy = ti.floor(x), ti.floor(y - 0.5)
    fx, fy = s - ix, t - iy
    # 边界插值使用镜像
    index00 = ti.Vector([ti.max(0, ti.min(res, int(ix))), ti.max(0, ti.min(res - 1, int(iy)))])
    index10 = ti.Vector([ti.max(0, ti.min(res, int(ix) + 1)), ti.max(0, ti.min(res - 1, int(iy)))])
    index01 = ti.Vector([ti.max(0, ti.min(res, int(ix))), ti.max(0, ti.min(res - 1, int(iy) + 1))])
    index11 = ti.Vector([ti.max(0, ti.min(res, int(ix) + 1)), ti.max(0, ti.min(res - 1, int(iy) + 1))])
    return bilerp(vx_field[index00], vx_field[index01], vx_field[index10], vx_field[index11], fx, fy)


@ti.func
def sample_vy(vy_field, coord):
    x, y = coord
    s, t = x - 0.5, y
    ix, iy = ti.floor(x - 0.5), ti.floor(y)
    fx, fy = s - ix, t - iy
    # 边界插值使用镜像
    index00 = ti.Vector([ti.max(0, ti.min(res - 1, int(ix))), ti.max(0, ti.min(res, int(iy)))])
    index10 = ti.Vector([ti.max(0, ti.min(res - 1, int(ix) + 1)), ti.max(0, ti.min(res, int(iy)))])
    index01 = ti.Vector([ti.max(0, ti.min(res - 1, int(ix))), ti.max(0, ti.min(res, int(iy) + 1))])
    index11 = ti.Vector([ti.max(0, ti.min(res - 1, int(ix) + 1)), ti.max(0, ti.min(res, int(iy) + 1))])
    return bilerp(vy_field[index00], vy_field[index01], vy_field[index10], vy_field[index11], fx, fy)


# 返回回溯之后的粒子位置
@ti.func
def backtrace(vx_field, vy_field, p, dt_):
    v1 = ti.Vector([sample_vx(vx_field, p), sample_vy(vy_field, p)])
    p1 = p - 0.5 * dt_ * v1
    v2 = ti.Vector([sample_vx(vx_field, p1), sample_vy(vy_field, p1)])
    p2 = p - 0.75 * dt_ * v2
    v3 = ti.Vector([sample_vx(vx_field, p2), sample_vy(vy_field, p2)])
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect_v(vx: ti.template(), new_vx: ti.template(), vy: ti.template(), new_vy: ti.template()):
    for i, j in vx:
        p = ti.Vector([i, j]) + ti.Vector([0, 0.5])
        p = backtrace(vx, vy, p, dt)
        new_vx[i, j] = sample_vx(vx, p)

    for i, j in vy:
        p = ti.Vector([i, j]) + ti.Vector([0.5, 0])
        p = backtrace(vx, vy, p, dt)
        new_vy[i, j] = sample_vy(vy, p)


@ti.kernel
def advect_c(f: ti.template(), f_new: ti.template(), vx: ti.template(), vy: ti.template()):
    for i, j in f:
        p = ti.Vector([i + 0.5, j + 0.5])
        p = backtrace(vx, vy, p, dt)
        f_new[i, j] = sample_p(f, p)


@ti.kernel
def apply_force(vx: ti.template(), vy: ti.template(), imp_data: ti.types.ndarray()):
    for i, j in vy:
        vy[i, j] -= 0 * dt

    origin_x, origin_y = imp_data[2], imp_data[3]
    dir = ti.Vector([imp_data[0], imp_data[1]])

    for i, j in vx:
        dx, dy = (i - origin_x), (j + 0.5 - origin_y)
        dr = dx * dx + dy * dy
        factor = ti.exp(-dr / force_radius)
        vx[i, j] += dir[0] * f_strength * factor * dt

    for i, j in vy:
        dx, dy = (i + 0.5 - origin_x), (j - origin_y)
        dr = dx * dx + dy * dy
        factor = ti.exp(-dr / force_radius)
        vy[i, j] += dir[1] * f_strength * factor * dt


class MouseDataGen:
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


@ti.kernel
def divergence(vx: ti.template(), vy: ti.template()):
    for i, j in ti.ndrange(res, res):
        vl = sample_vx(vx, ti.Vector([i, j + 0.5]))
        vr = sample_vx(vx, ti.Vector([i + 1, j + 0.5]))
        vt = sample_vy(vy, ti.Vector([i + 0.5, j + 1]))
        vb = sample_vy(vy, ti.Vector([i + 0.5, j]))

        if i == 0:
            vl = 0
        if i == res - 1:
            vr = 0
        if j == 0:
            vb = 0
        if j == res - 1:
            vt = 0

        velocity_divs[i, j] = vr - vl + vt - vb


@ti.kernel
def fill_laplacian_matrix(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(res, res):
        row = i * res + j
        center = 0.0
        if j != 0:
            A[row, row - 1] += -1.0
            center += 1.0
        if j != res - 1:
            A[row, row + 1] += -1.0
            center += 1.0
        if i != 0:
            A[row, row - res] += -1.0
            center += 1.0
        if i != res - 1:
            A[row, row + res] += -1.0
            center += 1.0
        A[row, row] += center


N = res * res
K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
F_b = ti.ndarray(ti.f32, shape=N)

fill_laplacian_matrix(K)
L = K.build()

solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(L)
solver.factorize(L)


@ti.kernel
def compute_Fb(div_in: ti.template(), div_out: ti.types.ndarray()):
    for i, j in ti.ndrange(res, res):
        div_out[j + i * res] = -div_in[i, j] / dt


Fb = ti.ndarray(ti.f32, shape=N)


@ti.kernel
def array_to_field(array_in: ti.types.ndarray(), field_out: ti.template()):
    for i, j in field_out:
        index = res * i + j
        field_out[i, j] = array_in[index]


@ti.kernel
def substract_gradient(vx: ti.template(), vy: ti.template(), pf: ti.template()):
    for i, j in vx:
        if i != 0 and i != res:
            pr = sample_p(pf, ti.Vector([i + 0.5, j + 0.5]))
            pl = sample_p(pf, ti.Vector([i - 0.5, j + 0.5]))
            vx[i, j] -= dt * (pr - pl)

    for i, j in vy:
        if j != 0 and j != res:
            pt = sample_p(pf, ti.Vector([i + 0.5, j + 0.5]))
            pb = sample_p(pf, ti.Vector([i + 0.5, j - 0.5]))
            vy[i, j] -= dt * (pt - pb)


def projection():
    divergence(velocities_x_pair.cur, velocities_y_pair.cur)
    compute_Fb(velocity_divs, Fb)
    x = solver.solve(Fb)
    array_to_field(x, pressures_pair.cur)
    substract_gradient(velocities_x_pair.cur, velocities_y_pair.cur, pressures_pair.cur)


def step(mouse_data):
    advect_v(velocities_x_pair.cur, velocities_x_pair.nxt, velocities_y_pair.cur, velocities_y_pair.nxt)
    advect_c(dyes_pair.cur,dyes_pair.nxt,velocities_x_pair.cur,velocities_y_pair.cur)
    velocities_x_pair.swap()
    velocities_y_pair.swap()
    dyes_pair.swap()
    apply_force(velocities_x_pair.cur, velocities_y_pair.cur, mouse_data)
    projection()


@ti.kernel
def v_visual(vx_f: ti.template(), vy_f: ti.template(), v: ti.template()):
    for i, j in v:
        pos = ti.Vector([i + 0.5, j + 0.5])
        vx = 0.5 * (vx_f[i, j] + vx_f[i + 1, j])
        vy = 0.5 * (vy_f[i, j] + vy_f[i, j + 1])
        norm = vx * vx + vy * vy
        v[i, j][0] = norm / (1 + norm)


def reset():
    velocities_x_pair.cur.fill(0)
    velocities_y_pair.cur.fill(0)
    pressures_pair.cur.fill(0)


def main():
    gui = ti.GUI("stable Fluid", (res, res))
    md_gen = MouseDataGen()
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == "r":
                reset()
        mouse_data = md_gen(gui)
        step(mouse_data)
        # divergence(velocities_x_pair.cur, velocities_y_pair.cur)
        # div_s = np.sum(velocity_divs.to_numpy())
        # print(f"divergence={velocity_divs[50, 50]}")

        gui.set_image(dyes_pair.cur)
        gui.show()


init()
main()
