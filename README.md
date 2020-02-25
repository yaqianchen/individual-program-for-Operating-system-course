# individual-program-for-Operating-system-course
five individual program for CSC3150, based on CPP, C and Cuda, for more detailed description, output, system environment about the project, see reports inside each folder

- file system simulation 
The program is written to simulate the file system based on contiguous allocagion. The program is implemented on CUDA and tested on the windows OS with CUDA version 9.2.148, VS version 2017, GPU for NVIDIA GeForce GTX 1060 6GB. The user should input the user program and a binary file, the program will automatically put all the data read into the snapshot.bin file.

- multi-process simulation
The program is written to implement the functions which is able to fork a child process to evoke the external program. The parent process is able to catch the signal raised in child process and return the value. The program is implemented on C programming language on ubuntu (ubuntu version 16.04 and kernel version v4.8.0-36-generic). The user should input the test external file and the output will be the signal raised in the file.
 
- prime-device simulation
The program is written to simulate the prime device in linux. The device can find the nth prime number and do the simple calculation. The program is implemented on C programming language on ubuntu (ubuntu version 16.04 and kernel version v4.8.0-36-generic). The user should input the test external file and the output will be the process of device control.

- pthread simulation(frog name)
The program is written to implement the frog game with the multithread method ----- pthread. The program contains two threads which control the frog move and the log move separately. The program is implemented on the cpp language and is executable on the ubuntu (ubuntu version 16.04 and kernel version v4.8.0-36-generic) and mac system. User should compile the cpp file and execute the executable file.
