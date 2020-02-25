#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){
    pid_t pid;
    pid = fork();
    int status;
    //if the fork failed
    if (pid == -1){
        perror("fork");
        exit(1);
    }
    else{
        /* fork a child process */
        if (pid == 0){
            char *arg[argc];
            /* execute test program */
            printf("I'm the Child process, my pid = %d\n",getpid());
            for (int i = 0; i < argc - 1; i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1] = NULL;
            printf("Child process start to execute the program\n");
            execve(arg[0],arg,NULL);
            perror("execve");
            //            raise(SIFCHILD);
        }
        else {
            /* wait for child process terminates */
            printf("I'm the Parent process, my pid = %d\n",getpid());
            waitpid(pid,&status,WUNTRACED);
            printf("Parent process receiving the SIGCHILD signal\n");
            /* check child process'  termination status */
// for exited signals
            if (WIFEXITED(status)){
                printf("Normal termination with EXIT STATUS = %d\n",status);
            }
// for signaled signals
            else if (WIFSIGNALED(status)){
                char* buff1, *buff2;
                if (WTERMSIG(status) == 1){
                    buff1 = "SIGHUP";
                    buff2 = "hangup";
                }
                else if (WTERMSIG(status) == 2){
                    buff1 = "SIGINT";
                    buff2 = "interrupt";
                }
                else if (WTERMSIG(status) == 3){
                    buff1 = "SIGQUIT";
                    buff2 = "quit";
                }
                else if (WTERMSIG(status) == 4){
                    buff1 = "SIGILL";
                    buff2 = "ill";
                }
                else if (WTERMSIG(status) == 5){
                    buff1 = "SIGTRAP";
                    buff2 = "trap";
                }
                else if (WTERMSIG(status) == 6){
                    buff1 = "SIGABRT";
                    buff2 = "abort";
                }
                else if (WTERMSIG(status) == 8){
                    buff1 = "SIGFPE";
                    buff2 = "floating point exception";
                }
                else if (WTERMSIG(status) == 9){
                    buff1 = "SIGKILL";
                    buff2 = "kill";
                }
                else if (WTERMSIG(status) == 7){
                    buff1 = "SIGBUS";
                    buff2 = "bus error";
                }
                else if (WTERMSIG(status) == 11){
                    buff1 = "SIGSEGV";
                    buff2 = "segment fault";
                }
                else if (WTERMSIG(status) == 13){
                    buff1 = "SIGPIPE";
                    buff2 = "broken pipe";
                }
                else if (WTERMSIG(status) == 14){
                    buff1 = "SIGALARM";
                    buff2 = "alarm";
                }
                else if (WTERMSIG(status) == 15){
                    buff1 = "SIGTERM";
                    buff2 = "terminate";
                }
                //                printf("wifsignal = %d\n",WTERMSIG(status));
                printf("child process get %s signal\n", buff1);
                printf("child process is %s by %s signal\n",buff2,buff2);
                printf("CHILD EXECUTION FAILED!!\n");
            }
// for stopped signals
            else if (WIFSTOPPED(status)){
                char *buff1, *buff2;
                if (WSTOPSIG(status) == 19){
                    buff1 = "SIGSTOP";
                    buff2 = "stopped";
                }
                printf("child process get %s signal\n", buff1);
                printf("child process %s\n", buff2);
                printf("CHILD PROCESS STOPPED\n");
            }
//if there is no valid signal returned
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
        }
    }
}
