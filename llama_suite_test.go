package llama_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestLLaMa(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "go-llama.cpp test suite")
}
