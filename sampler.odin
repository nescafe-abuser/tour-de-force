package tdf

import "core:slice"





Prob_Index :: struct {
	prob: f32,
	id:   i32,
}

Sampler :: struct {
	vocab_size:  i32,
	prob_index:  []Prob_Index,
	temperature: f32,
	topp:        f32,
	rng_state:   u64,
}






random_u32 :: proc(rng_state: ^u64) -> u32 {
	
	rng_state ^= rng_state^ >> 12
	rng_state ^= rng_state^ << 25
	rng_state ^= rng_state^ >> 27
	return u32((rng_state^ * 0x2545F4914F6CDD1D) >> 32)
}

random_f32 :: proc(rng_state: ^u64) -> f32 {
	return f32(random_u32(rng_state) >> 8) / 16777216.0
}





init_sampler :: proc(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Sampler {
	s: Sampler
	s.vocab_size = vocab_size
	s.temperature = temperature
	s.topp = topp

	
	s.rng_state = rng_seed if rng_seed != 0 else 1337

	s.prob_index = make([]Prob_Index, vocab_size)
	return s
}

free_sampler :: proc(s: ^Sampler) {
	delete(s.prob_index)
}






sample_argmax :: proc(probabilities: []f32) -> i32 {
	max_idx: i32 = 0
	max_val := probabilities[0]

	for i in 1 ..< len(probabilities) {
		if probabilities[i] > max_val {
			max_val = probabilities[i]
			max_idx = i32(i)
		}
	}

	return max_idx
}


sample_multinomial :: proc(probabilities: []f32, coin: f32) -> i32 {
	cdf: f32 = 0.0
	for i in 0 ..< len(probabilities) {
		cdf += probabilities[i]
		if coin < cdf {
			return i32(i)
		}
	}
	return i32(len(probabilities) - 1)
}


sample_topp :: proc(s: ^Sampler, logits: []f32, coin: f32) -> i32 {
	n := s.vocab_size
	cutoff := (1.0 - s.topp) / f32(n - 1)

	
	head := 0
	for i in 0 ..< n {
		if logits[i] >= cutoff {
			s.prob_index[head].id = i
			s.prob_index[head].prob = logits[i]
			head += 1
		}
	}

	
	valid_slice := s.prob_index[0:head]
	slice.sort_by(valid_slice, proc(a, b: Prob_Index) -> bool {
		return a.prob > b.prob
	})

	
	cumulative_prob: f32 = 0.0
	last_idx := head - 1

	for i in 0 ..< head {
		cumulative_prob += valid_slice[i].prob
		if cumulative_prob > s.topp {
			last_idx = i
			break
		}
	}

	
	r := coin * cumulative_prob
	cdf: f32 = 0.0

	for i in 0 ..= last_idx {
		cdf += valid_slice[i].prob
		if r < cdf {
			return valid_slice[i].id
		}
	}

	return valid_slice[last_idx].id
}


sample :: proc(s: ^Sampler, logits: []f32) -> i32 {
	
	if s.temperature == 0.0 {
		return sample_argmax(logits)
	}

	
	for i in 0 ..< len(logits) {
		logits[i] /= s.temperature
	}

	
	softmax(logits)

	
	coin := random_f32(&s.rng_state)

	if s.topp <= 0.0 || s.topp >= 1.0 {
		return sample_multinomial(logits, coin)
	} else {
		return sample_topp(s, logits, coin)
	}
}

