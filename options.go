package llama

type Options struct {
	ContextSize int
	NBatch      int
	VocabOnly   bool
	LowVRAM     bool
	Embeddings  bool
	NGPULayers  int

	Seed, Threads, Tokens, TopK, Repeat, Batch, NKeep int
	TopP, Temperature, Penalty                        float64
	F16KV                                             bool
	DebugMode                                         bool
	StopPrompts                                       []string
	IgnoreEOS                                         bool

	TailFreeSamplingZ float64
	TypicalP          float64
	FrequencyPenalty  float64
	PresencePenalty   float64
	Mirostat          int
	MirostatETA       float64
	MirostatTAU       float64
	PenalizeNL        bool
	LogitBias         string
	TokenCallback     func(string) bool

	PathPromptCache             string
	MLock, MMap, PromptCacheAll bool
	PromptCacheRO               bool
	MainGPU                     string
	TensorSplit                 string
}

type Option func(p *Options)

var DefaultOptions Options = Options{
	ContextSize:       512,
	Seed:              -1,
	MLock:             false,
	Embeddings:        false,
	MMap:              true,
	LowVRAM:           false,
	F16KV:             true,
	Threads:           4,
	Tokens:            128,
	Penalty:           1.1,
	Repeat:            64,
	Batch:             512,
	NKeep:             64,
	TopK:              40,
	TopP:              0.95,
	TailFreeSamplingZ: 1.0,
	TypicalP:          1.0,
	Temperature:       0.8,
	FrequencyPenalty:  0.0,
	PresencePenalty:   0.0,
	Mirostat:          0,
	MirostatTAU:       5.0,
	MirostatETA:       0.1,
}

// SetContext sets the context size.
func SetContext(c int) Option {
	return func(p *Options) {
		p.ContextSize = c
	}
}

func SetModelSeed(c int) Option {
	return func(p *Options) {
		p.Seed = c
	}
}

// SetContext sets the context size.
func SetMMap(b bool) Option {
	return func(p *Options) {
		p.MMap = b
	}
}

// SetNBatch sets the  n_Batch
func SetNBatch(n_batch int) Option {
	return func(p *Options) {
		p.NBatch = n_batch
	}
}

// Set sets the tensor split for the GPU
func SetTensorSplit(maingpu string) Option {
	return func(p *Options) {
		p.TensorSplit = maingpu
	}
}

// SetMainGPU sets the main_gpu
func SetMainGPU(maingpu string) Option {
	return func(p *Options) {
		p.MainGPU = maingpu
	}
}

// SetPredictionTensorSplit sets the tensor split for the GPU
func SetPredictionTensorSplit(maingpu string) Option {
	return func(p *Options) {
		p.TensorSplit = maingpu
	}
}

// SetPredictionMainGPU sets the main_gpu
func SetPredictionMainGPU(maingpu string) Option {
	return func(p *Options) {
		p.MainGPU = maingpu
	}
}

var VocabOnly Option = func(p *Options) {
	p.VocabOnly = true
}

var EnabelLowVRAM Option = func(p *Options) {
	p.LowVRAM = true
}

var EnableEmbeddings Option = func(p *Options) {
	p.Embeddings = true
}

var EnableF16KV Option = func(p *Options) {
	p.F16KV = true
}

var Debug Option = func(p *Options) {
	p.DebugMode = true
}

var EnablePromptCacheAll Option = func(p *Options) {
	p.PromptCacheAll = true
}

var EnablePromptCacheRO Option = func(p *Options) {
	p.PromptCacheRO = true
}

var EnableMLock Option = func(p *Options) {
	p.MLock = true
}

// Create a new Options object with the given options.
func NewOptions(opts ...Option) Options {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

var IgnoreEOS Option = func(p *Options) {
	p.IgnoreEOS = true
}

// SetMlock sets the memory lock.
func SetMlock(b bool) Option {
	return func(p *Options) {
		p.MLock = b
	}
}

// SetMemoryMap sets memory mapping.
func SetMemoryMap(b bool) Option {
	return func(p *Options) {
		p.MMap = b
	}
}

// SetGPULayers sets the number of GPU layers to use to offload computation
func SetGPULayers(n int) Option {
	return func(p *Options) {
		p.NGPULayers = n
	}
}

// SetTokenCallback sets the prompts that will stop predictions.
func SetTokenCallback(fn func(string) bool) Option {
	return func(p *Options) {
		p.TokenCallback = fn
	}
}

// SetStopWords sets the prompts that will stop predictions.
func SetStopWords(stop ...string) Option {
	return func(p *Options) {
		p.StopPrompts = stop
	}
}

// SetSeed sets the random seed for sampling text generation.
func SetSeed(seed int) Option {
	return func(p *Options) {
		p.Seed = seed
	}
}

// SetThreads sets the number of threads to use for text generation.
func SetThreads(threads int) Option {
	return func(p *Options) {
		p.Threads = threads
	}
}

// SetTokens sets the number of tokens to generate.
func SetTokens(tokens int) Option {
	return func(p *Options) {
		p.Tokens = tokens
	}
}

// SetTopK sets the value for top-K sampling.
func SetTopK(topk int) Option {
	return func(p *Options) {
		p.TopK = topk
	}
}

// SetTopP sets the value for nucleus sampling.
func SetTopP(topp float64) Option {
	return func(p *Options) {
		p.TopP = topp
	}
}

// SetTemperature sets the temperature value for text generation.
func SetTemperature(temp float64) Option {
	return func(p *Options) {
		p.Temperature = temp
	}
}

// SetPathPromptCache sets the session file to store the prompt cache.
func SetPathPromptCache(f string) Option {
	return func(p *Options) {
		p.PathPromptCache = f
	}
}

// SetPenalty sets the repetition penalty for text generation.
func SetPenalty(penalty float64) Option {
	return func(p *Options) {
		p.Penalty = penalty
	}
}

// SetRepeat sets the number of times to repeat text generation.
func SetRepeat(repeat int) Option {
	return func(p *Options) {
		p.Repeat = repeat
	}
}

// SetBatch sets the batch size.
func SetBatch(size int) Option {
	return func(p *Options) {
		p.Batch = size
	}
}

// SetKeep sets the number of tokens from initial prompt to keep.
func SetNKeep(n int) Option {
	return func(p *Options) {
		p.NKeep = n
	}
}

// SetTailFreeSamplingZ sets the tail free sampling, parameter z.
func SetTailFreeSamplingZ(tfz float64) Option {
	return func(p *Options) {
		p.TailFreeSamplingZ = tfz
	}
}

// SetTypicalP sets the typicality parameter, p_typical.
func SetTypicalP(tp float64) Option {
	return func(p *Options) {
		p.TypicalP = tp
	}
}

// SetFrequencyPenalty sets the frequency penalty parameter, freq_penalty.
func SetFrequencyPenalty(fp float64) Option {
	return func(p *Options) {
		p.FrequencyPenalty = fp
	}
}

// SetPresencePenalty sets the presence penalty parameter, presence_penalty.
func SetPresencePenalty(pp float64) Option {
	return func(p *Options) {
		p.PresencePenalty = pp
	}
}

// SetMirostat sets the mirostat parameter.
func SetMirostat(m int) Option {
	return func(p *Options) {
		p.Mirostat = m
	}
}

// SetMirostatETA sets the mirostat ETA parameter.
func SetMirostatETA(me float64) Option {
	return func(p *Options) {
		p.MirostatETA = me
	}
}

// SetMirostatTAU sets the mirostat TAU parameter.
func SetMirostatTAU(mt float64) Option {
	return func(p *Options) {
		p.MirostatTAU = mt
	}
}

// SetPenalizeNL sets whether to penalize newlines or not.
func SetPenalizeNL(pnl bool) Option {
	return func(p *Options) {
		p.PenalizeNL = pnl
	}
}

// SetLogitBias sets the logit bias parameter.
func SetLogitBias(lb string) Option {
	return func(p *Options) {
		p.LogitBias = lb
	}
}
