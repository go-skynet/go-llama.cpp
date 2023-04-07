# go-llama.cpp

[LLama.cpp](https://github.com/ggerganov/llama.cpp) golang bindings

Clone the repository locally:

```bash
git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp
```

To build the bindings locally, run:

```
cd go-llama.cpp
make libbinding.a
```

Then you can run the example with:

```
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/model/path/here" -t 14
```

Enjoy!

## License

MIT
