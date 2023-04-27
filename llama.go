package llama

// #cgo CXXFLAGS: -I./llama.cpp/examples -I./llama.cpp
// #cgo LDFLAGS: -L./ -lbinding -lm -lstdc++
// #cgo darwin LDFLAGS: -framework Accelerate
// #include "binding.h"
import "C"
import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

type LLama struct {
	state unsafe.Pointer
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	result := C.load_model(modelPath, C.int(mo.ContextSize), C.int(mo.Parts), C.int(mo.Seed), C.bool(mo.F16Memory), C.bool(mo.MLock))
	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	ll := &LLama{state: result}

	// set a finalizer to remove any callbacks when the struct is reclaimed by the garbage collector.
	runtime.SetFinalizer(ll, func(ll *LLama) {
		setCallback(ll.state, nil)
	})

	return ll, nil
}

func (l *LLama) Free() {
	C.llama_free_model(l.state)
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
	)
	ret := C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	for _, s := range po.StopPrompts {
		res = strings.TrimRight(res, s)
	}

	C.llama_free_params(params)

	return res, nil
}

// CGo only allows us to use static calls from C to Go, we can't just dynamically pass in func's.
// This is the next best thing, we register the callbacks in this map and call tokenCallback from
// the C code. We also attach a finalizer to LLama, so it will unregister the callback when the
// garbage collection frees it.

// SetTokenCallback registers a callback for the individual tokens created when running Predict. It
// will be called once for each token. The callback shall return true as long as the model should
// continue predicting the next token. When the callback returns false the predictor will return.
// The tokens are just converted into Go strings, they are not trimmed or otherwise changed. Also
// the tokens may not be valid UTF-8.
// Pass in nil to remove a callback.
//
// It is save to call this method while a prediction is running.
func (l *LLama) SetTokenCallback(callback func(token string) bool) {
	setCallback(l.state, callback)
}

var (
	m         sync.Mutex
	callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
	m.Lock()
	defer m.Unlock()

	if callback, ok := callbacks[uintptr(statePtr)]; ok {
		return callback(C.GoString(token))
	}

	return true
}

// setCallback can be used to register a token callback for LLama. Pass in a nil callback to
// remove the callback.
func setCallback(statePtr unsafe.Pointer, callback func(string) bool) {
	m.Lock()
	defer m.Unlock()

	if callback == nil {
		delete(callbacks, uintptr(statePtr))
	} else {
		callbacks[uintptr(statePtr)] = callback
	}
}
