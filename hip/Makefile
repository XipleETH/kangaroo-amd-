# HIP build for the MetaMask sweeper + kangaroo tools (ROCm / Linux).
# RX 7800 XT = gfx1101.  Usage:  make tests   |   make bip32_test   |   make clean
ARCH  ?= gfx1101
HIPCC ?= hipcc
FLAGS ?= -O3 --offload-arch=$(ARCH)

TESTS = sha_test bip39_test bip32_test metamask_bench

# generic: any <name> builds from <name>.hip
%: %.hip
	$(HIPCC) $(FLAGS) $< -o $@

.PHONY: tests clean
tests: $(TESTS)
	@for t in $(TESTS); do echo "== $$t =="; ./$$t || true; echo; done

clean:
	rm -f $(TESTS) pool_demo kangaroo field_test ec_test scalar_test sqr_test *.o
