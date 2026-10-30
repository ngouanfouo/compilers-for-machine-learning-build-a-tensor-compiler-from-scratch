"""
Compilers for Machine Learning: Build a Tensor Compiler from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Compilers for Machine Learning: a tensor compiler from scratch, measured against tinygrad.

Story: UOps and a rewrite engine simplify symbolic index math; movement ops become
index arithmetic pushed down to loads, so any tensor graph lowers to C loop nests
and a model splits into kernels. Loop transformations turn a naive GEMM into a
register-blocked one, and flash attention is a single fused reduce. The same IR
renders CUDA. Reverse-mode autodiff on the graph trains a tiny GPT entirely through
the compiler, tracking tinygrad's loss curve step for step.
"""
import math
import sys
import time
import numpy as np

sys.setrecursionlimit(100000)


def main() -> None:
    rng = np.random.default_rng(0)

    print("1. Symbolic rewrites")
    r0, r1, r2 = UOp.range(8, 0), UOp.range(4, 1), UOp.range(3, 2)
    flat = (r0 * 4 + r1) * 3 + r2
    e = (flat // 3) % 4
    print(f"   ((r0*4 + r1)*3 + r2) // 3 % 4 has {len(e.toposort())} nodes; after simplification: {len(simplify(e).toposort())} node ({'r1' if simplify(e) is r1 else '?'})")
    a, out = param("a", dtypes.float32, 1), param("out", dtypes.float32, 0)
    s = sink(store(out, 0, (load(a, 0) * 1.0 + 0.0) * 2.0 + (load(a, 1) * 0.0)))
    simp = sink(store(out, 0, simplify(s.src[0].src[1])))
    print(f"   loopless program: {render_sink(s).count(chr(10))} lines of C before rewriting, {render_sink(simp).count(chr(10))} after")

    print("\n2. Compile any model to C")
    xin = Tensor.input("x", (8, 16)); w1 = Tensor.input("w1", (16, 32)); b1 = Tensor.input("b1", (32,)); w2 = Tensor.input("w2", (32, 4))
    q = Tensor.input("wq", (16, 16)); k_ = Tensor.input("wk", (16, 16))
    h = (xin @ w1 + b1).relu()
    att = ((xin @ q) @ (xin @ k_).transpose()).softmax(-1) @ xin
    logits = ((h.layernorm() @ w2) + (att @ Tensor.input("wv", (16, 4)))).softmax(-1)
    prog = schedule({"y": logits})
    inputs = {"x": rng.standard_normal((8, 16)).astype(np.float32), "w1": rng.standard_normal((16, 32)).astype(np.float32) * 0.3,
              "b1": rng.standard_normal(32).astype(np.float32), "w2": rng.standard_normal((32, 4)).astype(np.float32) * 0.3,
              "wq": rng.standard_normal((16, 16)).astype(np.float32) * 0.3, "wk": rng.standard_normal((16, 16)).astype(np.float32) * 0.3,
              "wv": rng.standard_normal((16, 4)).astype(np.float32) * 0.3}
    t0 = time.time(); cp = Compiled(prog); ct = time.time() - t0
    got = cp.run(inputs)["y"]
    ref = eval_tensor(logits.uop, inputs)
    print(f"   MLP + attention block: {len(prog.kernels)} kernels ({cp.unique} unique sources), compiled in {ct:.2f}s, max |C - NumPy| = {float(np.abs(got - ref).max()):.1e}")
    img, wc = Tensor.input("img", (2, 3, 8, 8)), Tensor.input("wc", (4, 3, 3, 3))
    conv = img.conv2d(wc, pad=1).relu()
    kc = lower_kernel(conv.uop, {img.uop: "img", wc.uop: "wc"}, "conv")
    I, Wc = rng.standard_normal((2, 3, 8, 8)).astype(np.float32), rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    got = run_kernel_c(kc, [np.zeros((2, 4, 8, 8), np.float32), I, Wc])
    print(f"   3x3 convolution from movement ops: one kernel, {render_kernel(kc).count('for (')} loops, {len([u for u in kc.body[0].accs[0][2].toposort() if u.op is Ops.LOAD])} loads per step, max err {float(np.abs(got - eval_tensor(conv.uop, {'img': I, 'wc': Wc})).max()):.1e}")

    print("\n3. Fast kernels")
    n = 256
    A, B = Tensor.input("A", (n, n)), Tensor.input("B", (n, n))
    kg = lower_kernel((A @ B).uop, {A.uop: "A", B.uop: "B"}, "gemm")
    X, W = rng.standard_normal((n, n)).astype(np.float32), rng.standard_normal((n, n)).astype(np.float32)
    rows = []
    for label, kk in (("naive triple loop", kg), ("4x4 tile, K by 4", optimize_gemm(kg, 4, 4, 4)), ("4x8 tile, K by 8", optimize_gemm(kg, 4, 8, 8))):
        C = np.zeros((n, n), np.float32)
        t = bench_kernel(kk, [C, X, W])
        rows.append((label, gflops(n, t), float(np.abs(C - X @ W).max())))
    for label, g, err in rows:
        print(f"   {label:20s} {g:7.1f} GFLOP/s  (max err {err:.1e})")
    try:
        from tinygrad import Tensor as TT
        ta, tb = TT(X), TT(W)
        (ta @ tb).realize()
        t0 = time.perf_counter(); (ta @ tb).realize(); tt = time.perf_counter() - t0
        print(f"   {'tinygrad CPU backend':20s} {gflops(n, tt):7.1f} GFLOP/s  (its own codegen and search, same compiler)")
    except Exception as ex:
        print(f"   tinygrad comparison unavailable: {type(ex).__name__}")
    print(f"   speedup of the blocked kernel over the naive loop nest: {rows[2][1] / rows[0][1]:.1f}x")
    N, d = 128, 32
    Q, K_, V = (rng.standard_normal((N, d)).astype(np.float32) for _ in range(3))
    fk = flash_attention_kernel(N, d)
    S = Q @ K_.T / math.sqrt(d); P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); ref = P @ V
    O = np.zeros((N, d), np.float32)
    tf = bench_kernel(fk, [O, Q, K_, V])
    qt, kt, vt = Tensor.input("q", (N, d)), Tensor.input("k", (N, d)), Tensor.input("v", (N, d))
    unf = schedule({"o": ((qt @ kt.transpose()) * (1.0 / math.sqrt(d))).softmax(-1) @ vt})
    cpu = Compiled(unf)
    cpu.run({"q": Q, "k": K_, "v": V})
    t0 = time.perf_counter(); cpu.run({"q": Q, "k": K_, "v": V}); tu = time.perf_counter() - t0
    print(f"   flash attention (N={N}, d={d}): 1 fused kernel, {len(fk.body[0].accs)} accumulators, {1000*tf:.2f} ms, max err {float(np.abs(O - ref).max()):.1e}; unfused schedule: {len(unf.kernels)} kernels, {1000*tu:.2f} ms")

    print("\n4. GPU lowering")
    g = to_gpu(lower_kernel((Tensor.input("A", (64, 48)) @ Tensor.input("B", (48, 32))).uop, {Tensor.input("A", (64, 48)).uop: "A", Tensor.input("B", (48, 32)).uop: "B"}, "gemm"))
    grid, block = launch_dims(g, (32, 4, 1))
    src = render_cuda(g)
    a64, b48 = rng.standard_normal((64, 48)).astype(np.float32), rng.standard_normal((48, 32)).astype(np.float32)
    emu = run_kernel_np(g, [np.zeros((64, 32), np.float32), a64, b48])
    print(f"   64x48 @ 48x32 as a CUDA kernel: grid {grid}, block {block}, {len(src.splitlines())} lines, emulated max err {float(np.abs(emu - a64 @ b48).max()):.1e}")
    print("   " + src.splitlines()[0])
    print("   " + src.splitlines()[3].strip())

    print("\n5. Autodiff and a tiny GPT")
    x, wa, wb, y = Tensor.input("x", (4, 6)), Tensor.input("w1", (6, 8)), Tensor.input("w2", (8, 3)), Tensor.input("y", (4, 3))
    loss = -(((x @ wa).relu().layernorm() @ wb).logsoftmax(-1) * y).sum() * 0.25
    gr = backward(loss, [wa, wb])
    ins = {"x": rng.standard_normal((4, 6)).astype(np.float32), "w1": rng.standard_normal((6, 8)).astype(np.float32) * 0.5, "w2": rng.standard_normal((8, 3)).astype(np.float32) * 0.5}
    Y = np.zeros((4, 3), np.float32); Y[np.arange(4), rng.integers(0, 3, 4)] = 1; ins["y"] = Y
    an = eval_tensor(gr[wa].uop, ins)
    num = np.zeros_like(ins["w1"])
    for i in np.ndindex(num.shape):
        p = dict(ins); u = ins["w1"].copy(); u[i] += 1e-2; p["w1"] = u; lp = float(eval_tensor(loss.uop, p))
        u2 = ins["w1"].copy(); u2[i] -= 1e-2; p["w1"] = u2; lm = float(eval_tensor(loss.uop, p))
        num[i] = (lp - lm) / 2e-2
    print(f"   gradient of an MLP loss vs central differences: max |diff| = {float(np.abs(an - num).max()):.1e}")
    B, T, V, d = 4, 8, 16, 16
    res = train_gpt(B, T, V, d, steps=40, lr=0.5, seed=0)
    print(f"   GPT (V={V}, T={T}, d={d}): {res['kernels']} kernels per step ({res['unique']} unique), compiled in {res['compile_s']:.2f}s, {res['step_ms']:.1f} ms/step")
    print("   loss every 5 steps: " + " ".join(f"{l:.3f}" for l in res["losses"][::5]))
    try:
        from tinygrad import Tensor as TT
        from tinygrad.nn.optim import SGD
        trng = np.random.default_rng(0)
        p0 = gpt_init(V, T, d, trng)
        tp = {k: TT(v.copy()) for k, v in p0.items()}
        for t in tp.values(): t.requires_grad = True
        def tforward(xx, mask):
            Bn, Tn, Vn = xx.shape
            h0 = xx @ tp["wte"] + tp["wpe"]
            hh = h0.layernorm()
            qq, kk, vv = hh @ tp["wq"], hh @ tp["wk"], hh @ tp["wv"]
            sc = (qq.reshape(Bn, Tn, 1, d) * kk.reshape(Bn, 1, Tn, d)).sum(3) * (1.0 / math.sqrt(d)) + mask
            yy = (sc.softmax(-1).reshape(Bn, Tn, Tn, 1) * vv.reshape(Bn, 1, Tn, d)).sum(2)
            h0 = h0 + yy @ tp["wo"]
            h0 = h0 + ((h0.layernorm() @ tp["w1"]).relu() @ tp["w2"])
            return h0.layernorm() @ tp["wout"]
        opt = SGD(list(tp.values()), lr=0.5)
        tl = []
        for _ in range(10):
            xb, yb = make_batch(B, T, V, trng)
            with TT.train():
                tloss = -(tforward(TT(xb), TT(causal_mask(T))).log_softmax(-1) * TT(yb)).sum() * (1.0 / (B * T))
                tl.append(float(tloss.numpy()))
                opt.zero_grad(); tloss.backward(); opt.step()
        print("   tinygrad, same init and batches, first 10 losses: " + " ".join(f"{l:.3f}" for l in tl))
        print("   ours:                                             " + " ".join(f"{l:.3f}" for l in res["losses"][:10]))
        print(f"   max |ours - tinygrad| over 10 steps: {max(abs(a - b) for a, b in zip(tl, res['losses'])):.1e}")
    except Exception as ex:
        print(f"   tinygrad comparison unavailable: {type(ex).__name__}: {ex}")
    print("   The whole training step, forward, loss, backward and SGD, is a program of compiled C kernels produced by this compiler.")


if __name__ == "__main__":
    main()

