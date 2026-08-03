package tdf

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

Config :: struct #packed {
	dim:        i32, 
	hidden_dim: i32, 
	n_layers:   i32, 
	n_heads:    i32, 
	n_kv_heads: i32, 
	vocab_size: i32, 
	seq_len:    i32, 
}

Transformer_Weights :: struct {
	token_embedding_table: []f32, 
	rms_att_weight:        []f32,
	rms_ffn_weight:        []f32,
	wq:                    []f32, 
	wk:                    []f32,
	wv:                    []f32,
	wo:                    []f32,
	w1:                    []f32, 
	w2:                    []f32, 
	w3:                    []f32, 
	rms_final_weight:      []f32, 
	wcls:                  []f32, 
}


Run_State :: struct {
	x:           []f32, 
	xb:          []f32, 
	xb2:         []f32, 
	hb:          []f32, 
	hb2:         []f32, 
	q:           []f32, 
	k:           []f32, 
	v:           []f32, 
	att:         []f32, 
	logits:      []f32, 
	key_cache:   []f32, 
	value_cache: []f32, 
}

Transformer :: struct {
	config:    Config,
	weights:   Transformer_Weights,
	state:     Run_State,
	file_data: []byte, 
}





init_run_state :: proc(s: ^Run_State, p: ^Config) {
	kv_dim := (p.dim * p.n_kv_heads) / p.n_heads

	s.x = make([]f32, p.dim)
	s.xb = make([]f32, p.dim)
	s.xb2 = make([]f32, p.dim)
	s.hb = make([]f32, p.hidden_dim)
	s.hb2 = make([]f32, p.hidden_dim)
	s.q = make([]f32, p.dim)
	s.att = make([]f32, p.n_heads * p.seq_len)
	s.logits = make([]f32, p.vocab_size)
	s.key_cache = make([]f32, p.n_layers * p.seq_len * kv_dim)
	s.value_cache = make([]f32, p.n_layers * p.seq_len * kv_dim)
}

free_run_state :: proc(s: ^Run_State) {
	delete(s.x)
	delete(s.xb)
	delete(s.xb2)
	delete(s.hb)
	delete(s.hb2)
	delete(s.q)
	delete(s.att)
	delete(s.logits)
	delete(s.key_cache)
	delete(s.value_cache)
}

free_transformer :: proc(t: ^Transformer) {
	free_run_state(&t.state)
	if len(t.file_data) > 0 {
		delete(t.file_data)
	}
}

map_weights :: proc(w: ^Transformer_Weights, p: ^Config, ptr: ^f32, shared_weights: bool) {
	head_size := p.dim / p.n_heads
	n_layers := u64(p.n_layers)
	curr := ptr

	slice_and_advance :: proc(curr: ^^f32, count: u64) -> []f32 {
		s := mem.slice_ptr(curr^, int(count))
		curr^ = mem.ptr_offset(curr^, int(count))
		return s
	}

	w.token_embedding_table = slice_and_advance(&curr, u64(p.vocab_size) * u64(p.dim))
	w.rms_att_weight = slice_and_advance(&curr, n_layers * u64(p.dim))
	w.wq = slice_and_advance(&curr, n_layers * u64(p.dim) * u64(p.n_heads * head_size))
	w.wk = slice_and_advance(&curr, n_layers * u64(p.dim) * u64(p.n_kv_heads * head_size))
	w.wv = slice_and_advance(&curr, n_layers * u64(p.dim) * u64(p.n_kv_heads * head_size))
	w.wo = slice_and_advance(&curr, n_layers * u64(p.n_heads * head_size) * u64(p.dim))
	w.rms_ffn_weight = slice_and_advance(&curr, n_layers * u64(p.dim))
	w.w1 = slice_and_advance(&curr, n_layers * u64(p.dim) * u64(p.hidden_dim))
	w.w2 = slice_and_advance(&curr, n_layers * u64(p.hidden_dim) * u64(p.dim))
	w.w3 = slice_and_advance(&curr, n_layers * u64(p.dim) * u64(p.hidden_dim))
	w.rms_final_weight = slice_and_advance(&curr, u64(p.dim))

	curr = mem.ptr_offset(curr, int(p.seq_len * head_size))

	if shared_weights {
		w.wcls = w.token_embedding_table
	} else {
		w.wcls = slice_and_advance(&curr, u64(p.dim) * u64(p.vocab_size))
	}
}

init_transformer :: proc(checkpoint_path: string) -> (t: Transformer, ok: bool) {
	data, err := os.read_entire_file_from_path(checkpoint_path,context.allocator)
	if err != nil {
		fmt.eprintfln("Failed to open checkpoint file: %s", checkpoint_path)
		return t, false
	}
	t.file_data = data

	config_ptr := cast(^Config)raw_data(data)
	t.config = config_ptr^

	shared_weights := t.config.vocab_size > 0
	t.config.vocab_size = math.abs(t.config.vocab_size)

	init_run_state(&t.state, &t.config)

	weights_ptr := cast(^f32)mem.ptr_offset(raw_data(data), size_of(Config))
	map_weights(&t.weights, &t.config, weights_ptr, shared_weights)

	return t, true
}


forward :: proc(t: ^Transformer, token: i32, pos: i32) -> []f32 {
	p := &t.config
	w := &t.weights
	s := &t.state

	dim := int(p.dim)
	kv_dim := int((p.dim * p.n_kv_heads) / p.n_heads)
	kv_mul := int(p.n_heads / p.n_kv_heads)
	hidden_dim := int(p.hidden_dim)
	head_size := dim / int(p.n_heads)

	content_row := w.token_embedding_table[int(token) * dim:(int(token) + 1) * dim]
	copy(s.x, content_row)

	for l in 0 ..< int(p.n_layers) {

		rmsnorm(s.xb, s.x, w.rms_att_weight[l * dim:(l + 1) * dim])

		loff := l * int(p.seq_len) * kv_dim
		s.k = s.key_cache[loff + int(pos) * kv_dim:loff + (int(pos) + 1) * kv_dim]
		s.v = s.value_cache[loff + int(pos) * kv_dim:loff + (int(pos) + 1) * kv_dim]

		matmul(s.q, s.xb, w.wq[l * dim * dim:(l + 1) * dim * dim], dim, dim)
		matmul(s.k, s.xb, w.wk[l * dim * kv_dim:(l + 1) * dim * kv_dim], dim, kv_dim)
		matmul(s.v, s.xb, w.wv[l * dim * kv_dim:(l + 1) * dim * kv_dim], dim, kv_dim)

		for i := 0; i < dim; i += 2 {
			head_dim := i % head_size
			freq := 1.0 / math.pow(10000.0, f32(head_dim) / f32(head_size))
			val := f32(pos) * freq
			fcr := math.cos(val)
			fci := math.sin(val)

			rotn := 2 if i < kv_dim else 1
			for v_idx in 0 ..< rotn {
				vec := s.q if v_idx == 0 else s.k
				v0 := vec[i]
				v1 := vec[i + 1]
				vec[i] = v0 * fcr - v1 * fci
				vec[i + 1] = v0 * fci + v1 * fcr
			}
		}

		for h in 0 ..< int(p.n_heads) {
			q_head := s.q[h * head_size:(h + 1) * head_size]
			att_head := s.att[h * int(p.seq_len):(h + 1) * int(p.seq_len)]

			for timestep in 0 ..= int(pos) {
				k_head := s.key_cache[loff + timestep * kv_dim + (h / kv_mul) * head_size:]
				score: f32 = 0.0
				for i in 0 ..< head_size {
					score += q_head[i] * k_head[i]
				}
				score /= math.sqrt(f32(head_size))
				att_head[timestep] = score
			}

			softmax(att_head[0:pos + 1])

			xb_head := s.xb[h * head_size:(h + 1) * head_size]
			mem.zero_slice(xb_head)

			for timestep in 0 ..= int(pos) {
				v_head := s.value_cache[loff + timestep * kv_dim + (h / kv_mul) * head_size:]
				a := att_head[timestep]
				for i in 0 ..< head_size {
					xb_head[i] += a * v_head[i]
				}
			}
		}

		matmul(s.xb2, s.xb, w.wo[l * dim * dim:(l + 1) * dim * dim], dim, dim)

		for i in 0 ..< dim {
			s.x[i] += s.xb2[i]
		}

		rmsnorm(s.xb, s.x, w.rms_ffn_weight[l * dim:(l + 1) * dim])

		matmul(s.hb, s.xb, w.w1[l * dim * hidden_dim:(l + 1) * dim * hidden_dim], dim, hidden_dim)
		matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim:(l + 1) * dim * hidden_dim], dim, hidden_dim)

		for i in 0 ..< hidden_dim {
			val := s.hb[i]
			val *= (1.0 / (1.0 + math.exp(-val)))
			val *= s.hb2[i]
			s.hb[i] = val
		}

		matmul(s.xb, s.hb, w.w2[l * hidden_dim * dim:(l + 1) * hidden_dim * dim], hidden_dim, dim)

		for i in 0 ..< dim {
			s.x[i] += s.xb[i]
		}
	}

	rmsnorm(s.x, s.x, w.rms_final_weight)

	matmul(s.logits, s.x, w.wcls, dim, int(p.vocab_size))

	return s.logits
}

