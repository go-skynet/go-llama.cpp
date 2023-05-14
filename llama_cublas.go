//go:build cublas
// +build cublas

package llama

/*
#cgo LDFLAGS: -lcublas
*/
import "C"
