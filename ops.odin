package tdf

import "core:math"

rmsnorm :: proc(out: []f32, x: []f32, weight: []f32, eps: f32 = 1e-5) {
	size := len(x)

	ss: f32 = 0.0
	for j in 0 ..< size {
		ss += x[j] * x[j]
	}
	ss /= f32(size)
	ss += eps
	ss = 1.0 / math.sqrt(ss)

	
	for j in 0 ..< size {
		out[j] = weight[j] * (ss * x[j])
	}
}


softmax :: proc(x: []f32) {
	size := len(x)
	if size == 0 {
		return
	}

	
	max_val := x[0]
	for i in 1 ..< size {
		if x[i] > max_val {
			max_val = x[i]
		}
	}

	
	sum: f32 = 0.0
	for i in 0 ..< size {
		x[i] = math.exp(x[i] - max_val)
		sum += x[i]
	}

	
	for i in 0 ..< size {
		x[i] /= sum
	}
}



matmul :: proc(xout: []f32, x: []f32, w: []f32, n: int, d: int) {
	for i in 0 ..< d {
		val: f32 = 0.0
		w_row := w[i * n:(i + 1) * n]

		for j in 0 ..< n {
			val += w_row[j] * x[j]
		}
		xout[i] = val
	}
}



swiglu :: proc(hb: []f32, hb2: []f32) {
	size := len(hb)
	for i in 0 ..< size {
		val := hb[i]
		
		val *= (1.0 / (1.0 + math.exp(-val)))
		
		val *= hb2[i]
		hb[i] = val
	}
}


rope :: proc(q: []f32, k: []f32, pos: int, head_size: int, kv_dim: int) {
	dim := len(q)

	for i := 0; i < dim; i += 2 {
		head_dim := i % head_size
		freq := 1.0 / math.pow(10000.0, f32(head_dim) / f32(head_size))
		val := f32(pos) * freq
		fcr := math.cos(val)
		fci := math.sin(val)

		
		rotn := 2 if i < kv_dim else 1
		for v_idx in 0 ..< rotn {
			vec := q if v_idx == 0 else k
			v0 := vec[i]
			v1 := vec[i + 1]
			vec[i] = v0 * fcr - v1 * fci
			vec[i + 1] = v0 * fci + v1 * fcr
		}
	}
}

