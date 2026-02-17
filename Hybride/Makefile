MPICXX = mpic++
EXEC= main hybrideV1

all: $(EXEC)

main: main.o fonctions.o
	$(MPICXX) -o $@ $^ -fopenmp

hybrideV1: hybrideV1.o fonctions.o
	$(MPICXX) -o $@ $^ -fopenmp

%.o: %.cpp
	$(MPICXX) -o $@ -c $< -fopenmp

.PHONY: all clean mrproper

clean:
	@rm -rf *.o

mrproper: clean
	@rm -rf $(EXEC)
