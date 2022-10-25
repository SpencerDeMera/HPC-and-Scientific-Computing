#!/bin/bash

echo "Compile & Link HW2_Q1.cpp\n"
# Compile HW2_Q1.cpp
mpic++ -o Q1.out HW2_Q1.cpp

# ----- Run the program -----
mpirun -n 2 ./Q1.out

# ----- Program finished -----

echo "\nQuestion 1 Complete"

echo "\nCompile & Link HW2_Q2.cpp\n"
# Compile HW2_Q2.cpp
mpic++ -o Q2.out HW2_Q2.cpp

# ----- Run the program -----
echo "\n4 Processes:\n"
mpirun --oversubscribe -n 4 ./Q2.out
echo "\n8 Processes:\n"
mpirun --oversubscribe -n 8 ./Q2.out
echo "\n10 Processes:\n"
mpirun --oversubscribe -n 10 ./Q2.out
echo "\n12 Processes:\n"
mpirun --oversubscribe -n 12 ./Q2.out

# ----- Program finished -----

echo "\nQuestion 2 Complete"
echo "\nProgram Ending...\n"