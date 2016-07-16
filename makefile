CC=g++
FLAG=-std=c++11
PATH_SRC=./src/
PATH_MATH=./src/math/
PATH_TEST=./test/
DBN=$(PATH_SRC)dbn.h $(PATH_SRC)dbn.cc
RBM=$(PATH_SRC)rbm.h $(PATH_SRC)rbm.cc
MLP=$(PATH_SRC)mlp.h $(PATH_SRC)mlp.cc
MATH=$(PATH_MATH)*

all: mlp_test.o rbm_test.o dbn_test.o

dbn_test.o: $(MATH) $(RBM) $(MLP) $(DBN) $(PATH_TEST)dbn_test.cc
	$(CC) $(FLAG) $(MATH) $(RBM) $(MLP) $(DBN) $(PATH_TEST)dbn_test.cc -o dbn_test.o

rbm_test.o: $(MATH) $(RBM) $(PATH_TEST)rbm_test.cc
	$(CC) $(FLAG) $(MATH) $(RBM) $(PATH_TEST)rbm_test.cc -o rbm_test.o

mlp_test.o: $(MATH) $(MLP) $(PATH_TEST)mlp_test.cc
	$(CC) $(FLAG) $(MATH) $(MLP) $(PATH_TEST)mlp_test.cc -o mlp_test.o

clean:
	rm *.o