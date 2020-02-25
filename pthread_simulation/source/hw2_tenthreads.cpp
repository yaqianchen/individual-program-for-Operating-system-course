#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <iostream>
#include <random>

using namespace std;

#define ROW 10
#define COLUMN 50
#define NUM_THREADS 10

pthread_mutex_t frog_mutex;
pthread_cond_t frog_threshold_cv;
int count_num = 0;
int thread_ids[10] = {0,1,2,3,4,5,6,7,8,9};
int flag = 0;

struct Node{
    int x , y;
    Node( int _x , int _y ) : x( _x ) , y( _y ) {};
    Node(){} ;
} frog ;


char map[ROW+10][COLUMN] ;

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void){
    struct termios oldt, newt;
    int ch;
    int oldf;
    
    tcgetattr(STDIN_FILENO, &oldt);
    
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    
    ch = getchar();
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    
    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

//initialize the log
void initialize_the_log(){
    default_random_engine dre;
    uniform_int_distribution<int> di(0,COLUMN - 1);
    for (int i = 1; i < ROW; i++){
        int length = di(dre);
        for (int j = length; j < length + 17; j++){
            map[i][j % (COLUMN - 1)] = '=';
        }
    }
}

//check the status
int check_status(int x_after, int y_after){
    if (y_after >= COLUMN - 1 || y_after < 0) return -1;
    else if(x_after == 0) return 1;
    else if(x_after == ROW) return 0;
    else{
        if (map[x_after][y_after] == '='){
            return 0;
        }
        
        else return -1;
    }
}

//control the frog
void move_up(){
    if(frog.x == ROW){
        map[frog.x][frog.y] = '|';
    }
    else{
        map[frog.x][frog.y] = '=';
    }
    frog.x--;
    map[frog.x][frog.y] = '0' ;
}

void move_down(){
    if(frog.x == ROW){
        map[frog.x][frog.y] = '|';
    }
    else{
        map[frog.x][frog.y] = '=';
    }
    frog.x++;
    map[frog.x][frog.y] = '0' ;
}

void move_left(){
    if(frog.x == ROW){
        map[frog.x][frog.y] = '|';
    }
    else{
        map[frog.x][frog.y] = '=';
    }
    frog.y--;
    map[frog.x][frog.y] = '0' ;
}

void move_right(){
    if(frog.x == ROW){
        map[frog.x][frog.y] = '|';
    }
    else{
        map[frog.x][frog.y] = '=';
    }
    frog.y++;
    map[frog.x][frog.y] = '0' ;
}


//try_move
int try_move_up(){
    int x_after = frog.x - 1;
    int y_after = frog.y;
    return check_status(x_after,y_after);
}

int try_move_down(){
    int x_after = frog.x + 1;
    int y_after = frog.y;
    return check_status(x_after,y_after);
}

int try_move_left(){
    int x_after = frog.x;
    int y_after = frog.y - 1;
    return check_status(x_after,y_after);
}

int try_move_right(){
    int x_after = frog.x;
    int y_after = frog.y + 1;
    return check_status(x_after,y_after);
}

//control the log
void *logs_move( void *t ){
    int* my_id = (int*)t;
    
    while (!flag){
        usleep(300000);
        pthread_mutex_lock(&frog_mutex);
        /*  Move the logs  */
        if ((*my_id + 1) % 2 != 0){
            for (int j = 0; j < COLUMN - 1; j++){
                if (map[*my_id + 1][j] == '=') {
                    map[*my_id + 1][(j + 48) % (COLUMN - 1) ] = '=';
                    map[*my_id + 1][j] = ' ';
                }
                else if(map[*my_id + 1][j] == '0'){
                    map[*my_id + 1][(j + 48) % (COLUMN - 1) ] = '0';
                    map[*my_id + 1][j] = ' ';
                    frog.y --;
                    if (frog.y >= COLUMN - 1 || frog.y < 0) flag = 1;
                }
            }
        }
        else{
            for (int j = COLUMN - 2; j > -1; j--){
                if (map[*my_id + 1][j] == '=') {
                    map[*my_id + 1][(j + 1) % (COLUMN - 1)] = '=';
                    map[*my_id + 1][j] = ' ';
                }
                else if(map[*my_id + 1][j] == '0'){
                    map[*my_id + 1][(j + 1) % (COLUMN - 1)] = '0';
                    map[*my_id + 1][j] = ' ';
                    frog.y++;
                    if (frog.y >= COLUMN - 1 || frog.y < 0) flag = 1;
                }
            }
        }
        count_num += 1;
        if (count_num == NUM_THREADS-1) pthread_cond_signal(&frog_threshold_cv);
        pthread_mutex_unlock(&frog_mutex);
    }
    pthread_exit(NULL);
    return 0;
}


void *frog_move( void *t ){
    
    int* my_id = (int*)t;
    while (!flag){
        pthread_mutex_lock(&frog_mutex);
        while (count_num != NUM_THREADS-1) pthread_cond_wait(&frog_threshold_cv,&frog_mutex);
        count_num = 0;
        
        /*  Check keyboard hits, to change frog's position or quit the game. */
        if (kbhit()){
            char dir = getchar();
            if ( dir == 'w' || dir == 'W') {
                if (try_move_up() == 1 || try_move_up() == -1){
                    frog.x--;
                    flag = 1;
                }
                else move_up();
            }
            if ( dir == 's' || dir == 'S') {
                if (frog.x != ROW){
                    if (try_move_down() == 1 || try_move_down() == -1){
                        frog.x++;
                        flag = 1;
                    }
                    else move_down();
                }
            }
            if ( dir == 'a' || dir == 'A') {
                if (try_move_left() == 1 || try_move_left() == -1){
                    frog.y--;
                    flag = 1;
                }
                else move_left();
            }
            if ( dir == 'd' || dir == 'D') {
                if (try_move_right() == 1 || try_move_right() == -1){
                    frog.y++;
                    flag = 1;
                }
                else move_right();
            }
            if ( dir == 'q' || dir == 'Q') {
                flag = 1;
            }
        }
        puts("\033[H\033[2J");
        for(int i = 0; i <= ROW; ++i){
            puts( map[i] );
        }
        pthread_mutex_unlock(&frog_mutex);
    }
    pthread_exit(NULL);
    return 0;
}

int main( int argc, char *argv[] ){
    
    // Initialize the river map and frog's starting position
    memset( map , 0, sizeof( map ) ) ;
    int i , j ;
    
    //initialize the map
    for( i = 1; i < ROW; ++i ){
        for( j = 0; j < COLUMN - 1; ++j )
            map[i][j] = ' ' ;
    }
    
    //initialize both the sides along the river
    for( j = 0; j < COLUMN - 1; ++j )
        map[ROW][j] = map[0][j] = '|' ;
    
    //initialize the frog positionin the middle
    frog = Node( ROW, (COLUMN-1) / 2 ) ;
    map[frog.x][frog.y] = '0' ;
    
    //initialize the log position
    initialize_the_log();
    
    //Print the map into screen
    for( i = 0; i <= ROW; ++i)
        puts( map[i] );
    
    pthread_t threads[10];
    pthread_attr_t attr;
    
    //initialize the threads
    pthread_mutex_init(&frog_mutex,NULL);
    pthread_cond_init(&frog_threshold_cv,NULL);
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0],&attr,logs_move,(void*)&thread_ids[0]);
    pthread_create(&threads[1],&attr,logs_move,(void*)&thread_ids[1]);
    pthread_create(&threads[2],&attr,logs_move,(void*)&thread_ids[2]);
    pthread_create(&threads[3],&attr,logs_move,(void*)&thread_ids[3]);
    pthread_create(&threads[4],&attr,logs_move,(void*)&thread_ids[4]);
    pthread_create(&threads[5],&attr,logs_move,(void*)&thread_ids[5]);
    pthread_create(&threads[6],&attr,logs_move,(void*)&thread_ids[6]);
    pthread_create(&threads[7],&attr,logs_move,(void*)&thread_ids[7]);
    pthread_create(&threads[8],&attr,logs_move,(void*)&thread_ids[8]);
    pthread_create(&threads[9],&attr,frog_move,(void*)&thread_ids[9]);
    
    for (int i=0; i<NUM_THREADS; i++){
        pthread_join(threads[i],NULL);
    }
    
    //display the game information
    puts("\033[H\033[2J");
    if (check_status(frog.x,frog.y) == 1){
        cout << "You Win" << endl;
    }
    else{
        cout << "You Lose" << endl;
    }
    
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&frog_mutex);
    pthread_cond_destroy(&frog_threshold_cv);
    
    pthread_exit(NULL);
    return 0;
}
