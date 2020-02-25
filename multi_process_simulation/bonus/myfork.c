#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

int pid_array[100];
int signal_array[100];
int count = 0;

void execution_recursion (int index, int numInput, char* arg[])
{
    pid_t pid;
    pid = vfork();
    int status;
    //if the fork failed
    if (pid == -1){
        perror("fork");
        exit(1);
    }
    else{
        if (pid == 0){
            //child process
            if (index == numInput){
                //for debug
                //printf("index for the last = %d\n",index);
                execve(arg[numInput - 1],arg,NULL);
            }
            else{
                //printf("for other children\n");
                execution_recursion(++index,numInput,arg);
                //execve(arg[index - 1],arg,NULL);
            }
        }
        else {
            waitpid(pid,&status,WUNTRACED);
            //for debug
	    if (count != 0){
	    index--;
}
	    count++;
	    //printf("count = %d",count);
            //printf("index for the parent = %d\n",index);
           // printf("status = %d\n",status);
            pid_array[index - 1] = pid;
            signal_array[index - 1] = status;

            if (index != 1){
                execve(arg[index - 2],arg,NULL);
            }
        }
    }
}

void display(){
    printf("the pid_array = ");
    for (int i = 0; i < 4; i++){
        printf("%d : %d  ",i,pid_array[i]);
    }
    printf("\n");
    printf("the signal_array = ");
    for (int i = 0; i < 4; i++){
        printf("%d : %d  ",i, signal_array[i]);
    }
    printf("\n");
}

int main(int argc,char *argv[])
{
    char *arg[argc];
    for (int i = 0; i < argc - 1; i++){
        arg[i]=argv[i+1];
    }
    arg[argc-1] = NULL;
    
    //check the invalid input
    if (argc <= 1){
        printf("Invalid input\n");
	exit(1);
    }
    
    //do the recursion
    int initial = 1;
    execution_recursion(initial, argc - 1, arg);
    
    //check the array (for debug)
    //display();
    
    //process tree
    printf("The process tree: %d", getpid());
    for (int i = 0; i < argc - 1; i++){
        printf("->%d",pid_array[i]);
    }
    printf("\n");
    
    //show the child and parent process info
    int num_previous, num_after;

    for (int i = 0;i <= argc - 2; i++){
        if (i == argc - 2){
            num_previous = argc - 2 - i;
            num_after = getpid();
        }
        else{
            num_previous = argc - 2 - i;
            num_after = pid_array[argc - 3 - i];
        }

        // normal
        if (signal_array[num_previous] == 0){
            printf("The child process (pid=%d) of parent process (pid=%d) has normal execution\n",pid_array[num_previous],num_after);
            printf("Its exit status = 0\n\n");
        }
        else if (signal_array[num_previous] == 19){
            printf("The child process (pid=%d) of parent process (pid=%d) is stopped by signal\n",pid_array[num_previous],num_after);
            printf("Its signal number is %d\n",num_previous);
            printf("Child process got SIGSTOP signal\n");
            printf("Child process stopped\n\n");
        }
        else{
            char *buff1, *buff2;
            // SIGHUP
            if (signal_array[num_previous] == 1){
                buff1 = "SIGHUP";
                buff2 = "hang up";
            }
            // SIGINT
            else if (signal_array[num_previous] == 2){
                buff1 = "SIGINT";
                buff2 = "interrupt";
            }
            /* SIGQUIT */
            else if (signal_array[num_previous]== 3){
                buff1 = "SIGQUIT";
                buff2 = "quit";
            }
            /* SIGILL */
            else if (signal_array[num_previous] == 4){
                buff1 = "SIGILL";
                buff2 = "illegal instruction";
            }
            /* SIGTRAP */
            else if (signal_array[num_previous] == 5){
                buff1 = "SIGTRAP";
                buff2 = "trap";
            }
            /* SIGABRT */
            else if (signal_array[num_previous] == 6){
                buff1 = "SIGABRT";
                buff2 = "abort";
            }
            /* SIGBUS */
            else if (signal_array[num_previous] == 7){
                buff1 = "SIGBUS";
                buff2 = "bus";
            }
            /* SIGFPE */
            else if (signal_array[num_previous] == 8){
                buff1 = "SIGFPE";
                buff2 = "floating point exception";
            }
            /* SIGKILL */
            else if (signal_array[num_previous] == 9){
                buff1 = "SIGKILL";
                buff2 = "kill";
            }
            /* SIGSEGV */
            else if (signal_array[num_previous] == 11){
                buff1 = "SIGSEGV";
                buff2 = "segment fault";
            }
            /* SIGPIPE */
            else if (signal_array[num_previous] == 13){
                buff1 = "SIGPIPE";
                buff2 = "pipe";
            }
            /* SIGALRM */
            else if (signal_array[num_previous] == 14){
                buff1 = "SIGALRM";
                buff2 = "alrm";
            }
            /* SIGTERM */
            else if (signal_array[num_previous] == 15){
                buff1 = "SIGTERM";
                buff2 = "termination";
            }
            printf("The child process (pid=%d) of parent process (pid=%d) is stopped by signal\n",pid_array[num_previous],num_after);
            printf("Its signal number is %d\n",signal_array[num_previous]);
            printf("Child process got %s signal\n",buff1);
            printf("Child was terminated by %s signal\n\n",buff2);
        }
    }
    printf("Myfork process(pid=%d) execute normally\n",getpid());
    return 0;
}
