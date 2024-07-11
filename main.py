import math
import time

import numpy as np
import pickle

from typing import List

import taichi as ti
import taichi.math as tm
from PIL import Image
import os
import sys


def sketch_decimation(v,Nout=50):
    N = len(v)
    A = np.zeros((N))
    for i in range(N):
        j = (i+1)%N
        k = (i+2)%N
        A[j] = v[i,0]*v[j,1] + v[j,0]*v[k,1] + v[k,0]*v[i,1]\
             - v[i,0]*v[k,1] - v[j,0]*v[i,1] - v[k,0]*v[j,1]

    A = np.abs(A)
    return v[np.sort(np.argpartition(A, -Nout)[-Nout:])]


def load_sketch(path):
    with open(path, "rb") as f:
        cad = pickle.load(f)

    n_sketch = len(cad['sketch'])
    # print(f"n_sketch: {n_sketch}")
    # print(cad['sketch'])
    sketches = []
    nv_max = 0
    for i in range(n_sketch):
        # v_ = sketch_decimation(cad['sketch'][1].astype(np.float32))
        v_ = cad['sketch'][i]
        if v_ is None:
            continue
        v_ = v_.astype(np.float32)
        nv_ = len(v_)
        nv_max = max(nv_max, nv_)
        sketches.append((v_,nv_))

    # print(f"nv_max: {nv_max}")
    v  = ti.Vector.field(n=2, dtype=ti.f32, shape=(n_sketch, nv_max))
    nv = ti.field(dtype=ti.i32, shape=(n_sketch))
    print(f"nv_max: {nv_max}")

    vnp = np.zeros((n_sketch, nv_max,2), dtype=np.float32)
    for i,(v_,nv_) in enumerate(sketches):
        vnp[i] = np.pad(v_, ((0,nv_max-nv_),(0,0)), 'constant', constant_values=(0))/2
        nv[i] = nv_
    v.from_numpy(vnp)

    h = ti.field(dtype=ti.f32, shape=(n_sketch))             # height
    s = ti.field(dtype=ti.f32, shape=(n_sketch))             # scale
    c = ti.Vector.field(n=3, dtype=ti.f32, shape=(n_sketch)) # centroid
    a = ti.Vector.field(n=3, dtype=ti.f32, shape=(n_sketch)) # axis

    h.from_torch(cad['height'])
    s.from_torch(cad['scale'])
    c.from_torch(cad['centroid'])
    a.from_torch(cad['axis'])

    o = ti.field(ti.f32, shape=(n_sketch)) # operation (-1 == complement)
    o.fill(0.0)
    return n_sketch, v, nv, h, s, c, a, o


@ti.func
def sdf_polygon(p,v,nv,idx):
    d = tm.dot(p-v[idx,0],p-v[idx,0])
    s = 1.0
    j = nv[idx]-1
    for i in range(nv[idx]):
        e = v[idx,j] - v[idx,i]
        w =        p - v[idx,i]
        b = w - e*tm.clamp(tm.dot(w,e)/tm.dot(e,e), 0.0, 1.0)
        d = tm.min(d, tm.dot(b,b))
        c = (p[1] >= v[idx,i].y) + (p[1] < v[idx,j].y) + (e.x*w.y>e.y*w.x)
        if (c == 3.0 or c == 0.0): s *= -1.0;
        j = i
    return s * tm.sqrt(d)

@ti.func
def extrusion_sdf(p,v,nv,h,s,c,a,idx):
    scale = 0.5
    wx = sdf_polygon((p[:2]-c[idx].xy)/s[idx]*scale,v,nv,idx)
    wy = ti.abs((p[2]-c[idx].z)*scale) - ti.f32(h[idx]/2.0)
    return tm.min(tm.max(wx,wy),0.0) + tm.sqrt(tm.max(wx,0.0)**2+tm.max(wy,0.0)**2);

@ti.func
def op_sdf_subtract(d1,d2):
    return ti.max(-d1,d2)

@ti.func
def op_sdf_union(d1,d2):
    return ti.min(d1,d2)

@ti.func
def sdf(p):
    curr_d = extrusion_sdf(p,V,NV,H,S,C,A,I[0])
    inst_idx = 0
    for j in range(1,N):
        i = I[j]
        if ti.int32(O[i]) == 0:
            new_d = extrusion_sdf(p,V,NV,H,S,C,A,i)
            if new_d == op_sdf_union(curr_d, new_d):
                inst_idx = j
            curr_d = op_sdf_union(curr_d, new_d)
        else:
            curr_d = op_sdf_subtract(extrusion_sdf(p,V,NV,H,S,C,A,i), curr_d)
    return curr_d, inst_idx

@ti.func
def ray_march(p, d, R):
    j = 0
    dist = 0.0
    inst_i = -1
    while j < 100 and sdf(R @ (p + dist * d))[0] > 1e-6 and dist < inf:
        d_val, inst_i = sdf(R @ (p + dist * d))
        dist += d_val
        j += 1
    return ti.min(inf, dist), inst_i

@ti.func
def sdf_normal(p, R):
    d = 1e-3
    n = ti.Vector([0.0, 0.0, 0.0])
    sdf_center = sdf(R @ p)[0]
    for i in ti.static(range(3)):
        inc = p
        inc[i] += d
        n[i] = (1 / d) * (sdf(R @ inc)[0] - sdf_center)
    return n.normalized()


@ti.func
def next_hit(pos, d, R):
    closest, normal, c = inf, ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
    ray_march_dist, inst_i = ray_march(pos, d, R)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
        normal = sdf_normal(pos + d * closest, R)
    return closest, normal, inst_i

@ti.kernel
def render(R: ti.types.matrix(3, 3, ti.f32)):
    R_ = R.inverse()
    for u,v in normal_buffer:
        pos = camera_pos
        aspect_ratio = res[0] / res[1]
        d = ti.Vector(
            [
                2 * fov * (u+ti.random()) / res[1] - fov * aspect_ratio - 1e-5,
                2 * fov * (v+ti.random()) / res[1] - fov - 1e-5,
                -1.0,
            ]
        )
        d = d.normalized()

        depth = 0.0
        normal = ti.Vector([0.0, 0.0, 1.0])

        closest, normal, _ = next_hit(pos, d, R_)

        depth_buffer[u,v] = closest
        normal_buffer[u,v] = normal

@ti.func
def shade_matcap(hit_normal, ray_dir, col):
    up_dir = ti.Vector([0.0,1.0,0.0])
    ray_up   = (up_dir - tm.dot(up_dir, ray_dir) * up_dir).normalized()
    ray_left = tm.cross(ray_dir, ray_up)
    matcap_u = tm.dot(-ray_left, hit_normal)
    matcap_v = tm.dot(ray_up, hit_normal)

    matcap_v *= 0.95
    matcap_u *= 0.95

    matcap_i = ti.cast(ti.round((matcap_u+1.0)  / 2.0 * matcap_img.shape[0]), ti.i32)
    matcap_j = ti.cast(ti.round((-matcap_v+1.0) / 2.0 * matcap_img.shape[1]), ti.i32)

    return matcap_img[matcap_i, matcap_j] * col

@ti.kernel
def render_hires(R: ti.types.matrix(3, 3, ti.f32)):
    R_ = R.inverse()
    max_iter = 2
    for u, v in color_buffer_hires:
        depth: ti.f32 = 0.0
        normal = ti.Vector([0.0,0.0,0.0])
        pos = camera_pos
        aspect_ratio = hires[0] / hires[1]
        d = ti.Vector(
            [
                2 * fov * (u+ti.random()) / hires[1] - fov * aspect_ratio - 1e-5,
                2 * fov * (v+ti.random()) / hires[1] - fov - 1e-5,
                -1.0,
            ]
        )
        d = d.normalized()

        inst_idx = 0
        for i in range(max_iter):
            depth, normal_, inst_idx = next_hit(pos, d, R_)
            normal += normal_

        normal /= max_iter
        color_buffer_hires[u,v].w = 0.0
        if depth < inf:
            col = shade_matcap(normal, d, color_map[inst_idx])
            #if inst_idx  >= 0:
            # col = color_map[inst_idx]
            color_buffer_hires[u,v].xyz = col 
            color_buffer_hires[u,v].w = 255.0


# def main():
if __name__ == "__main__":
    import time
    mouse_prv = None
    path = str(sys.argv[1])

    ti.init(arch=ti.gpu)
    res = 1024//3, 1024//3
    depth_buffer = ti.Vector.field(1, dtype=ti.f32, shape=res)
    normal_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

    hires = 1024//3, 1024//3
    color_buffer_hires = ti.Vector.field(4, dtype=ti.f32,  shape=hires)
    matcap_np = np.array(Image.open("./matcap2.png"), dtype=np.float32)[...,1]
    matcap_np = ((255.0 - matcap_np)/255.0 + 0.3).clip(0,255.0)
    matcap_np = matcap_np * 0.8
    matcap_np = matcap_np.astype(np.uint8)
    matcap_img = ti.field(ti.f32, shape=matcap_np.shape)
    matcap_img.from_numpy(matcap_np)

    color_map = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
    color_map.from_numpy(np.array([ [176, 197, 164], [211, 118, 118],
                                    [235, 196, 159], [155, 176, 193],
                                    [173, 136, 198], [81, 130, 155],
                                    [180, 180, 184], [58, 77, 57],], dtype=np.float32))
    # color_map = {0: ti.Vector(n=3, dtype=ti.u8).from_numpy((numpy.array([176, 197, 164], dtype=np.uint8))), 
    #              1:(211, 118, 118), 2:(235, 196, 159), 3:(155, 176, 193), 4:(173, 136, 198), 5:(81, 130, 155), 6:(180, 180, 184), 7:(58, 77, 57),8:(169, 68, 56)}

    max_ray_depth = 6
    eps = 1e-4
    inf = 1e10

    fov = 0.23
    dist_limit = 1000

    camera_pos = ti.Vector([0.0, 0.0, 10.0])

    scene_rotation = np.array([[ 9.99915004e-01, -7.83732720e-03 ,-1.04481932e-02],
                            [-5.35510480e-09, -7.99960315e-01 , 6.00054145e-01],
                            [-1.30610615e-02, -6.00003064e-01, -7.99892366e-01]])
    scene_rotation = ti.Matrix(scene_rotation)

    # pickle_pathes = os.listdir("./fusion_test_pickle")
    # os.makedirs("./fusion_test_vis")
    # import gc


    name = path.split(".")[0]
    print(name)

    # path = os.path.join("./fusion_mv2cyl_test", path)
    print(path)

    N, V, NV, H, S, C, A, O = load_sketch(path)
    I = ti.field(ti.i32, shape=N)
    I.from_numpy(np.array(range(N)).astype(np.uint8))
    NUM_SUBTRACT = 0

    render_hires(scene_rotation)
    Image.fromarray(color_buffer_hires.to_numpy().astype(np.uint8).swapaxes(0,1)).save(f'./{name}_new.png')

    print(f"done {name}")
    # gc.collect()


    # uncomment for offline gui
    # import polyscope as ps
    # import polyscope.imgui as psim
    # ps.init()
    # ps.set_ground_plane_mode("none")

    # def callback():
    #     nonlocal scene_rotation
    #     global O,NUM_SUBTRACT
    #     p = np.array(json.loads(ps.get_view_as_json())['viewMat']).reshape(4,4)[:3,:3].astype(np.float32)
    #     scene_rotation.from_numpy(p.T)
    #     render(scene_rotation)
    #     depth = depth_buffer.to_numpy().swapaxes(0,1).squeeze()
    #     normal = normal_buffer.to_numpy().swapaxes(0,1)
    #     ps.add_depth_render_image_quantity("render_img", depth, normal,
    #                                        enabled=True, image_origin='lower_left',
    #                                        color=(1.0, 1.0, 1.0), material='wax', transparency=1.0,
    #                                        allow_fullscreen_compositing=False)


    #     if (psim.Button("render hi-res")):
    #         render_hires(scene_rotation)
    #         print("scene_roatation", scene_rotation.to_numpy())
    #         print("cam_pos", camera_pos)
    #         Image.fromarray(color_buffer_hires.to_numpy().swapaxes(0,1)).save('out.png')
    #         print("done")


    #     for i in range(N):
    #         is_changed, Oi = psim.Checkbox(f"subtract {i}", O[i])

    #         if (is_changed):
    #             if Oi == 0:
    #                 NUM_SUBTRACT += 1
    #             else:
    #                 NUM_SUBTRACT -= 1

    #             O[i] = Oi
    #             I.from_numpy(np.argsort(O.to_numpy()))

    # ps.set_user_callback(callback)
    # ps.show()

# if __name__ == "__main__":
    # main()
