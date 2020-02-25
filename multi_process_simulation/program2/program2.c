#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

static struct task_struct* task;

struct wait_opts {
    enum pid_type wo_type; //It is defined in ‘/include/linux/pid.h’.
    int wo_flags; //Wait options. (0, WNOHANG, WEXITED, etc.)
    struct pid *wo_pid;  //Kernel's internal notion of a process identifier. “Find_get_pid()”
    struct siginfo __user *wo_info; //Singal information.
    int __user *wo_stat; // Child process’s termination status
    struct rusage __user *wo_rusage; //Resource usage
    wait_queue_t child_wait; //Task wait queue
    int notask_error;};

extern long  do_wait (struct wait_opts *wo);

extern int do_execve (
                      struct filename *filename,
                      const char __user *const __user *__argv,
                      const char __user *const __user *__envp);

extern long _do_fork(
                     unsigned long clone_flags,
                     unsigned long stack_start,
                     unsigned long stack_size,
                     int __user *parent_tidptr,
                     int __user *child_tidptr,
                     unsigned long tls);

extern struct filename *getname(const char __user *filename);

//int my_fork(void *argc);
//int my_exec(void);
//void my_wait(pid_t pid);

int my_exec(void){
    printk("[program2] : child process");
    int result;
    const char path[] = "/home/seed/work/source/program2/test";
    const char *const argv[] = {path,NULL,NULL};
    const char *const envp[] = {"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};
    
    struct filename * my_filename = getname(path);
    
    result = do_execve(my_filename,argv,envp);
    
    if(!result){
        return 0;
    }
    
    do_exit(result);
}

void my_wait(pid_t pid){
    int a;
    int status;
    struct wait_opts wo;
    struct pid *wo_pid = NULL;
    enum pid_type type;
    type = PIDTYPE_PID;
    wo_pid = find_get_pid(pid);
    
    wo.wo_type = type;
    wo.wo_pid = wo_pid;
    wo.wo_flags = WEXITED;
    wo.wo_info = NULL;
    wo.wo_stat = (int __user*)&status;
    wo.wo_rusage = NULL;
    
    a = do_wait(&wo);
    
    if (*wo.wo_stat == 17){
        printk("[program2] : Normal termination with EXIT STATUS = %d\n",*wo.wo_stat);
    }
    // for stopped
    else if (*wo.wo_stat == 19){
        char *buff1;
        buff1 = "SIGSTOP";
        printk("[program2] : get %s signal\n", buff1);
        printk("[program2] : CHILD PROCESS STOPPED\n");
    }
    // for signaled
    else {
        char* buff1, *buff2;
        if (*wo.wo_stat == 1){
            buff1 = "SIGHUP";
            buff2 = "hangup";
        }
        else if (*wo.wo_stat == 2){
            buff1 = "SIGINT";
            buff2 = "interrupt";
        }
        else if (*wo.wo_stat == 3){
            buff1 = "SIGQUIT";
            buff2 = "quit";
        }
        else if (*wo.wo_stat == 4){
            buff1 = "SIGILL";
            buff2 = "illegal instruction";
        }
        else if (*wo.wo_stat == 5){
            buff1 = "SIGTRAP";
            buff2 = "trap";
        }
        else if (*wo.wo_stat == 6){
            buff1 = "SIGABRT";
            buff2 = "abort";
        }
        else if (*wo.wo_stat == 7){
            buff1 = "SIGBUS";
            buff2 = "bus error";
        }
        else if (*wo.wo_stat == 8){
            buff1 = "SIGFPE";
            buff2 = "floating point exception";
        }
        else if (*wo.wo_stat == 9){
            buff1 = "SIGKILL";
            buff2 = "kill";
        }
        else if (*wo.wo_stat == 11){
            buff1 = "SIGSEGV";
            buff2 = "segment fault";
        }
        else if (*wo.wo_stat == 13){
            buff1 = "SIGPIPE";
            buff2 = "broken pipe";
        }
        else if (*wo.wo_stat == 14){
            buff1 = "SIGALARM";
            buff2 = "alarm";
        }
        else if (*wo.wo_stat == 15){
            buff1 = "SIGTERM";
            buff2 = "terminate";
        }
        else{
            printk("[program2] : process continues");
        }
        //                printk("wifsignal = %d\n",WTERMSIG(status));
        printk("[program2] : get %s signal\n", buff1);
        printk("[program2] : child process has %s error\n",buff2);
    }
    
    printk("[program2] : The return signal is %d\n",*wo.wo_stat);
    put_pid(wo_pid);
    return;
}

//implement fork function
int my_fork(void *argc){
    //set default sigaction for current process
    int i;
    pid_t pid;
    struct k_sigaction *k_action = &current->sighand->action[0];
    for(i=0;i<_NSIG;i++){
        k_action->sa.sa_handler = SIG_DFL;
        k_action->sa.sa_flags = 0;
        k_action->sa.sa_restorer = NULL;
        sigemptyset(&k_action->sa.sa_mask);
        k_action++;
    }
    /* fork a process using do_fork */
    pid = _do_fork(SIGCHLD,(unsigned long)&my_exec,0,NULL,NULL,0);
    /* execute a test program in child process */
    printk("[program2] : The child process has pid = %d\n",pid);
    printk("[program2] : This is the parent process,pid = %d\n",(int)current->pid);
    /* wait until child process terminates */
    my_wait(pid);
    return 0;
}

static int __init program2_init(void){
    
    printk("[program2] : module_init\n");
    
    /* write your code here */
    task = kthread_create(&my_fork,NULL,"Mythread");
    printk("[program2] : module_init create kthread start\n");
    
    if(!IS_ERR(task)){
        printk("[program2] : module_init Kthread starts\n");
        wake_up_process(task);
    }
    /* create a kernel thread to run my_fork */
    return 0;
}

static void __exit program2_exit(void){
    printk("[program2] : module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);

