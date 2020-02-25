#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define DIR = 0;
#define FILE = 1;

using namespace std;

__device__ __managed__ u32 gtime = 1;
__device__ __managed__ u32 gtime_create = 1;
__device__ __managed__ u32 block_position = 0;
__device__ __managed__ u32 FCB_position = 4096;
__device__ __managed__ u32 current_FCB_position = 4096;

__device__ void display(FileSystem* fs) {
	printf("this fs////////////////////////////////////////////");
	printf("in the zero position\n");
	printf("name = %s\n", fs->directory[0].name);
	printf("sibling = %d\n", fs->directory[0].sibling);
	printf("child = %d\n", fs->directory[0].child);
	printf("parent = %d\n", fs->directory[0].parent);
	printf("size = %d\n", fs->directory[0].size);
	printf("indentity = %d\n", fs->directory[0].identity);
	printf("create time = %d\n", fs->directory[0].create_date);
	printf("modified time = %d\n", fs->directory[0].modified_date);

	printf("\n");

	printf("in the one position\n");
	printf("name = %s\n", fs->directory[1].name);
	printf("sibling = %d\n", fs->directory[1].sibling);
	printf("parent = %d\n", fs->directory[1].parent);
	printf("child = %d\n", fs->directory[1].child);
	printf("size = %d\n", fs->directory[1].size);
	printf("indentity = %d\n", fs->directory[1].identity);
	printf("create time = %d\n", fs->directory[1].create_date);
	printf("modified time = %d\n", fs->directory[1].modified_date);

	printf("\n");

	printf("in the second position\n");
	printf("name = %s\n", fs->directory[2].name);
	printf("sibling = %d\n", fs->directory[2].sibling);
	printf("parent = %d\n", fs->directory[2].parent);
	printf("child = %d\n", fs->directory[2].child);
	printf("size = %d\n", fs->directory[2].size);
	printf("indentity = %d\n", fs->directory[2].identity);
	printf("create time = %d\n", fs->directory[2].create_date);
	printf("modified time = %d\n", fs->directory[2].modified_date);

	printf("\n");

	printf("in the third position\n");
	printf("name = %s\n", fs->directory[3].name);
	printf("sibling = %d\n", fs->directory[3].sibling);
	printf("parent = %d\n", fs->directory[3].parent);
	printf("child = %d\n", fs->directory[3].child);
	printf("size = %d\n", fs->directory[3].size);
	printf("indentity = %d\n", fs->directory[3].identity);
	printf("create time = %d\n", fs->directory[3].create_date);
	printf("modified time = %d\n", fs->directory[3].modified_date);
}

__device__ void display_valid(FileSystem* fs) {
	printf("this is valid fs//////////////////////////////////");
	printf("in the zero position\n");
	printf("name = %s\n", fs->valid_directory[0].name);
	printf("sibling = %d\n", fs->valid_directory[0].sibling);
	printf("child = %d\n", fs->valid_directory[0].child);
	printf("parent = %d\n", fs->valid_directory[0].parent);
	printf("size = %d\n", fs->valid_directory[0].size);
	printf("indentity = %d\n", fs->valid_directory[0].identity);
	printf("create time = %d\n", fs->valid_directory[0].create_date);
	printf("modified time = %d\n", fs->valid_directory[0].modified_date);

	printf("\n");

	printf("in the one position\n");
	printf("name = %s\n", fs->valid_directory[1].name);
	printf("sibling = %d\n", fs->valid_directory[1].sibling);
	printf("parent = %d\n", fs->valid_directory[1].parent);
	printf("child = %d\n", fs->valid_directory[1].child);
	printf("size = %d\n", fs->valid_directory[1].size);
	printf("indentity = %d\n", fs->valid_directory[1].identity);
	printf("create time = %d\n", fs->valid_directory[1].create_date);
	printf("modified time = %d\n", fs->valid_directory[1].modified_date);

	printf("\n");

	printf("in the second position\n");
	printf("name = %s\n", fs->valid_directory[2].name);
	printf("sibling = %d\n", fs->valid_directory[2].sibling);
	printf("parent = %d\n", fs->valid_directory[2].parent);
	printf("child = %d\n", fs->valid_directory[2].child);
	printf("size = %d\n", fs->valid_directory[2].size);
	printf("indentity = %d\n", fs->valid_directory[2].identity);
	printf("create time = %d\n", fs->valid_directory[2].create_date);
	printf("modified time = %d\n", fs->valid_directory[2].modified_date);

	printf("\n");

	printf("in the third position\n");
	printf("name = %s\n", fs->valid_directory[3].name);
	printf("sibling = %d\n", fs->valid_directory[3].sibling);
	printf("parent = %d\n", fs->valid_directory[3].parent);
	printf("child = %d\n", fs->valid_directory[3].child);
	printf("size = %d\n", fs->valid_directory[3].size);
	printf("indentity = %d\n", fs->valid_directory[3].identity);
	printf("create time = %d\n", fs->valid_directory[3].create_date);
	printf("modified time = %d\n", fs->valid_directory[3].modified_date);
}



__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;
  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
}

__device__ __managed__ int current_index = 0;
__device__ __managed__ int last_index = 1;
__device__ __managed__ int current_depth = 0;
__device__ __managed__ int write_index = 0;

__device__ void segment_management_FCB(FileSystem *fs, u32 fp) {
	for (int i = fp; i < 36863; i = i + 32) {
		if (fs->volume[i + 32] == 0 && fs->volume[i + 32 + 1] == 0 && fs->volume[i + 32 + 2] == 0 && fs->volume[i + 32 + 3] == 0) break;
		for (int j = 0; j < 32; j++) {
			fs->volume[i + j] = fs->volume[i + j + 32];
			fs->volume[i + j + 32] = 0;
		}
	}
}

__device__ void segment_management(FileSystem *fs, u32 fp, u32 original_size) {

	//manage the volume
	u32 position = fs->FILE_BASE_ADDRESS + fp * 32;
	u32 size = ((original_size - 1) / 32 + 1) * 32;
	while ((fs->volume[position + size] != 0 || (position + size) %32 != 0)&& position + original_size < fs->STORAGE_SIZE) {
		fs->volume[position] = fs->volume[position + size];
		fs->volume[position + size] = 0;
		position++;
	}

	//manage the block
	for (int i = 0; i < block_position / 8 + 1; i++) {
		fs->volume[i] = 0;
	}
	block_position = block_position - (original_size - 1) / 32 - 1;
	u32 whole_block = block_position / 8;
	u32 remainder = block_position % 8;

	for (int i = 0; i < whole_block && i < fs->SUPERBLOCK_SIZE ; i++) {
		fs->volume[i] = 511;
	}
	for (int i = 0; i < remainder; i++) {
		fs->volume[whole_block] = fs->volume[whole_block] + (1 << i);
	}

	//change FCB
	u32 FCB_block_position;
	for (int i = 4096; i < 36863; i = i + 32) {
		if (fs->volume[i] == 0 && fs->volume[i + 1] == 0 && fs->volume[i + 2] == 0 && fs->volume[i + 3] == 0) break;
		FCB_block_position = (fs->volume[i + 28] << 24) + (fs->volume[i + 29] << 16) + (fs->volume[i + 30] << 8) + (fs->volume[i + 31]);
		if (FCB_block_position > fp) {
			FCB_block_position = FCB_block_position - (original_size - 1) / 32 - 1;
			fs->volume[i + 28] = FCB_block_position >> 24;
			fs->volume[i + 29] = FCB_block_position >> 16;
			fs->volume[i + 30] = FCB_block_position >> 8;
			fs->volume[i + 31] = FCB_block_position;
		}
	}
}

__device__ void display(FileSystem*fs, u32 stop_position, int op) {
	//display date
	if (op == 0) {
		printf("stop position = %d\n", stop_position);
		printf("===sort by modified time===\n");
		for (u32 i = 0; i <= stop_position; i++) {
			if (fs->valid_directory[i].identity == 0) printf("%s d\n", fs->valid_directory[i].name);
			else printf("%s\n",fs->valid_directory[i].name);
		}
	}
	else {
		printf("stop position = %d\n", stop_position);
		printf("===sort by file size===\n");
		for (u32 i = 0; i <= stop_position; i++) {
			if (fs->valid_directory[i].identity == 0) printf("%s %d d\n", fs->valid_directory[i].name, fs->valid_directory[i].size);
			else printf("%s %d\n", fs->valid_directory[i].name, fs->valid_directory[i].size);
		}
	}
}

__device__ void swap(FileSystem* fs, int x, int y) {
	struct file_directory tempt = fs->valid_directory[x];
	fs->valid_directory[x] = fs->valid_directory[y];
	fs->valid_directory[y] = tempt;
}

__device__ void bubblesort(FileSystem *fs, u32 left, u32 right, int op) {

	// sort by date
	if (op == 0) {
		for (int i = left; i < right; i ++) {
			for (int j = left; j < right - i + left; j++) {
				int j_date_previous = fs->valid_directory[j].modified_date;
				int j_date_after = fs->valid_directory[j+1].modified_date;
				if (j_date_previous < j_date_after) swap(fs, j, j + 1);
			}
		}
	}
	else {
		for (int i = left; i < right; i++) {
			for (int j = left; j < right - i + left; j++) {
				int j_size_previous = fs->valid_directory[j].size;
				int j_size_after = fs->valid_directory[j + 1].size;
				int j_date_previous = fs->valid_directory[j].create_date;
				int j_date_after = fs->valid_directory[j + 1].create_date;
				if (j_size_previous < j_size_after) swap(fs, j, j + 1);
				if (j_size_after == j_size_previous && j_date_previous > j_date_after) swap(fs, j, j + 1);
			}
		}
	}
}

__device__ u32 if_exist(FileSystem *fs, char *s) {
	//return FCB position
	int flag;
	for (int i = 4096; i < 36863; i = i + 32) {
		flag = 0;
		if (fs->volume[i] == 0 && fs->volume[i + 1] == 0 && fs->volume[i + 2] == 0 && fs->volume[i + 3] == 0) {
			break;
		}
		for (int j = 4; j < 24; j++) {
			if (fs->volume[i + j] != s[j - 4]) {
				flag = 1;
				break;
			}
		}
		if (flag == 1) continue;
		if (flag == 0) return i;
	}
	return -1;
}

__device__ bool check_the_name(char*name1, char*name2) {
	for (int i = 0; i < 20; i++) {
		if (name1[i] != name2[i]) return true;
	}
	return false;
}

__device__ int if_exist_directory(FileSystem *fs, char *s) {
	if (fs->directory[current_index].child == NULL) return -1;
	else {
		int directory_index = fs->directory[current_index].child;
		while (check_the_name(fs->directory[directory_index].name, s) && fs->directory[directory_index].sibling != NULL) directory_index = fs->directory[directory_index].sibling;
		if (check_the_name(fs->directory[directory_index].name, s)) return -directory_index;
		return directory_index;
	}
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	printf("//////////////////////////////////////////before open\n");
	display(fs);
	/* Implement open operation here */
	//if not exist
	int check = if_exist_directory(fs, s);
	//printf("check in open is %d\n", check);
	if (check < 0) {
		printf("file do not exist\n");
		if (op == 0) {
			printf("can not find the file to read error\n");
			return -1;
		}
		//store in the directory
		int name_count = 0;
		fs->directory[last_index].child = NULL;
		fs->directory[last_index].identity = 1;
		for (int i = 0; i < 20 && (i == 0 || s[i - 1] != '\0'); i++) {
			fs->directory[last_index].name[i] = s[i];
			name_count++;
		}
		fs->directory[last_index].parent = current_index;
		fs->directory[last_index].sibling = NULL;
		fs->directory[last_index].size = 0;
		fs->directory[last_index].create_date = gtime_create;
		fs->directory[last_index].modified_date = gtime;
		if (fs->directory[current_index].child == NULL) fs->directory[current_index].child = last_index;
		else fs->directory[-check].sibling = last_index;
		fs->directory[current_index].size += name_count;
		write_index = last_index;
		last_index++;

	//implement the volume
		//store the name
		current_FCB_position = FCB_position;
		//printf("for name in open = ");
		for (int i = 4; i < 24; i++) {
			fs->volume[FCB_position + i] = s[i - 4];
		}

		//store the create date
		fs->volume[FCB_position + 24] = gtime_create >> 8;
		fs->volume[FCB_position + 25] = gtime_create;

		//strore the modified date 
		fs->volume[FCB_position + 26] = gtime >> 8;
		fs->volume[FCB_position + 27] = gtime;

		//store the start block
		fs->volume[FCB_position + 28] = block_position >> 24;
		fs->volume[FCB_position + 29] = block_position >> 16;
		fs->volume[FCB_position + 30] = block_position >> 8;
		fs->volume[FCB_position + 31] = block_position;

		//update the date
		gtime++;
		gtime_create++;

		//update FCB position
		FCB_position = FCB_position + 32;
		return block_position;
	}

	//if exist
	else {
		current_FCB_position = if_exist(fs, s);
		u32 start_block = (fs->volume[current_FCB_position + 28] << 24) + (fs->volume[current_FCB_position + 29] << 16) + (fs->volume[current_FCB_position + 30] << 8) + (fs->volume[current_FCB_position + 31]);
	
		//if write
		if (op == 1) {
			write_index = check;
			fs->directory[write_index].modified_date = gtime;
			//clean the old file in volume
			u32 size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
			for (int i = 0; i < size; i++) {
				fs->volume[start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
			}

			//clean the old file in block
			for (int i = 0; i < (size - 1) / 32 + 1; i++) {
				u32 super_block_position = start_block + i;
				int shift_number = super_block_position % 8;
				fs->volume[super_block_position / 8] = fs->volume[super_block_position / 8] - (1 << shift_number);
			}

			//update FCB date
			fs->volume[current_FCB_position + 26] = gtime >> 8;
			fs->volume[current_FCB_position + 27] = gtime;

			//update the date
			gtime++;
		}
		return start_block;
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	printf("///////////////////////////////////before read\n");
	display(fs);
	/* Implement read operation here */
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	printf("///////////////////////////////////////////before write\n");
	display(fs);
	/* Implement write operation here */
    //enough space
	fs->directory[write_index].size = size;
	if ((fs->volume[(fp + (size - 1) / 32)/8] >> (fp + (size - 1) / 32) % 8) % 2 == 0) {
		u32 old_file_size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
		u32 original_size = old_file_size - size;

		//update volume
		for (int i = 0; i < size; i++) {
			fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];

		//update block
			if (i % 32 == 0) { 
				u32 super_block_position = fp + i / 32;
				int shift_number = super_block_position % 8;
				fs->volume[(fp + i /32) / 8] = fs->volume[(fp + i / 32) / 8] + (1 << shift_number);
			}
		}
		if (int (original_size) < 0) block_position = block_position + (-original_size - 1) / 32 + 1;

		//update size
		fs->volume[current_FCB_position] = size >> 24;
		fs->volume[current_FCB_position + 1] = size >> 16;
		fs->volume[current_FCB_position + 2] = size >> 8;
		fs->volume[current_FCB_position + 3] = size;
		if (original_size > 0 && old_file_size != 0 && fp != block_position - 1) segment_management(fs, fp + (size - 1) / 32 + 1, original_size);
	}

	//out of space
	else {
		u32 original_size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
		if (block_position * 32 - 1 + size >= fs->SUPERBLOCK_SIZE) {
			return -1;
		}

		//update volume
		else {
			for (int i = 0; i < size; i++) { 
				fs->volume[block_position * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];

				//update block
				if (i % 32 == 0) {
					u32 super_block_position = block_position + i / 32;
					int shift_number = super_block_position % 8;
					fs->volume[(block_position + i / 32) / 8] = fs->volume[(block_position + i / 32) / 8] + (1 << shift_number);
				}
			}

			//update size
			fs->volume[current_FCB_position] = size >> 24;
			fs->volume[current_FCB_position + 1] = size >> 16;
			fs->volume[current_FCB_position + 2] = size >> 8;
			fs->volume[current_FCB_position + 3] = size;

			//update block position
			fs->volume[current_FCB_position + 28] = block_position >> 16;
			fs->volume[current_FCB_position + 29] = block_position >> 16;
			fs->volume[current_FCB_position + 30] = block_position >> 8;
			fs->volume[current_FCB_position + 31] = block_position;
		}
		segment_management(fs, fp, original_size);
	}
}

__device__ int morrisTraversal(FileSystem* fs, int root_index) {
	int count = 0;
	int parent_index = root_index;
	while (fs->directory[parent_index].sibling != NULL) {
		if (fs->directory[parent_index].child != NULL) {
			int child_index = fs->directory[parent_index].child;
			while (fs->directory[child_index].sibling != NULL) {
				if (fs->directory[child_index].child != NULL) {
					int grand_child_index = fs->directory[child_index].child;
					while (fs->directory[grand_child_index].sibling != NULL) {
						fs->valid_directory[count] = fs->directory[grand_child_index];
						printf("count = %d,  %d\n", count, grand_child_index);
						count++;
						grand_child_index = fs->directory[grand_child_index].sibling;
					}
					fs->valid_directory[count] = fs->directory[grand_child_index];
					printf("count = %d,  %d\n", count, child_index);
					count++;
				}
				fs->valid_directory[count] = fs->directory[child_index];
				printf("count = %d,  %d\n", count, child_index);
				count++;
				child_index = fs->directory[child_index].sibling;
			}
			fs->valid_directory[count] = fs->directory[child_index];
			printf("count = %d,  %d\n", count, child_index);
			count++;
		}
		fs->valid_directory[count] = fs->directory[parent_index];
		printf("count = %d,  %d\n", count, parent_index);
		count++;
		parent_index = fs->directory[parent_index].sibling;
	}
	fs->valid_directory[count] = fs->directory[parent_index];
	printf("count = %d,  %d\n", count, parent_index);
	count++;
	return count;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	if (op == 0 || op == 1) {
		int count = morrisTraversal(fs, fs->directory[0].child);
		printf("/////////////////////after morrisTraversal\n");
		display(fs);
		display_valid(fs);
		printf("///////////////////////////count is %d\n", count);
		bubblesort(fs, 0, count - 1, op);
		printf("///////////////////////////after bubble sort\n");
		display(fs);
		display_valid(fs);
		display(fs, count - 1, op);
	}
	// CD_P
	else if (op == 6) {
		//no parent
		if (fs->directory[current_index].parent == NULL) printf("no parent error\n");
		else current_index = fs->directory[current_index].parent;
	}
	//PWD
	else if (op == 5) {
		int index_directory = current_index;
		if (current_depth == 1) printf("/%s\n", fs->directory[current_index].name);
		else if (current_depth == 2) printf("/%s/%s\n", fs->directory[fs->directory[current_index].parent].name, fs->directory[current_index].name);
		else {
			int parent_index = fs->directory[current_index].parent;
			char* parent = fs->directory[parent_index].name;
			int pp_index = fs->directory[parent_index].parent;
			char* parent_parent = fs->directory[pp_index].name;
			printf("/%s/%s/%s\n", parent_parent, parent, fs->directory[current_index].name);
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if (op == 2) {
		int index_previous = -1;
		if (fs->directory[current_index].child == NULL) printf("no subdirectory error\n");
		int index_directory = fs->directory[current_index].child;
		while (fs->directory[index_directory].name != s && fs->directory[index_directory].sibling != NULL) {
			index_previous = index_directory;
			index_directory = fs->directory[index_directory].sibling;
		}
		if (fs->directory[index_directory].name != s) printf("no such directory error\n");
		//clean the file inside
		// it is file
		if (fs->directory[index_directory].identity == 1) {
			//rm operation for directory tree 

			/* Implement rm operation here */
			if (if_exist(fs, s) == -1) printf("no such file founded error\n");
			else {
				current_FCB_position = if_exist(fs, s);

				//change volume
				u32 start_block = (fs->volume[current_FCB_position + 28] << 24) + (fs->volume[current_FCB_position + 29] << 16) + (fs->volume[current_FCB_position + 30] << 8) + (fs->volume[current_FCB_position + 31]);

				//clean the old file in volume
				u32 size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
				for (int i = 0; i < size; i++) {
					fs->volume[start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
				}

				//clean the old file in block
				for (int i = 0; i < (size - 1) / 32 + 1; i++) {
					fs->volume[start_block + i] = 0;
				}

				//clean the FCB
				for (int i = 0; i < 32; i++) {
					fs->volume[current_FCB_position + i] = 0;
				}
				segment_management(fs, start_block, size);
				segment_management_FCB(fs, current_FCB_position);
				FCB_position = FCB_position - 32;
			}
			if (index_previous != -1) fs->directory[index_previous].sibling = NULL;
			else fs->directory[current_index].child = NULL;
		}
		else printf("can not use RM to remove the directory\n");
	}
	else if (op == 3) {
		//MKDIR
		//for debug
		printf("///////////////////////////////////in MKDIR\n");
		display(fs);
		if (last_index > 1024) printf("file out of storage error\n");
		//for debug
		printf("last_index is %d\n", last_index);
		if (fs->directory[current_index].identity == 1) printf("can not MKDIR in file error\n");
		if (current_depth >= 3) printf("file out of depth\n");
		int index_directory = fs->directory[current_index].child;
		printf("index_directory = %d\n", index_directory);

		//no other file
		if (index_directory == NULL) {
			printf("no other file in MKDIR\n");
			int name_count = 0;
			for (int i = 0; i < 20 || s[i - 1] != '\0'; i++) {
				fs->directory[last_index].name[i] = s[i];
				name_count++;
			}
			fs->directory[last_index].sibling = NULL;
			fs->directory[last_index].child = NULL;
			fs->directory[last_index].parent = current_index;
			fs->directory[last_index].identity = 0;
			fs->directory[last_index].size = 0;
			fs->directory[last_index].modified_date = gtime;
			fs->directory[last_index].create_date = gtime_create;
			fs->directory[current_index].child = last_index;
			fs->directory[current_index].size += name_count;
		}
		//other files
		else {
			printf("other file\n");
			int file_count = 0;
			int name_count = 0;
			while (fs->directory[index_directory].sibling != NULL) {
				index_directory = fs->directory[index_directory].sibling;
				file_count++;
			}
			printf("index_directory = %d\n", index_directory);
			if (file_count >= 50) printf("file out of directory storage\n");
			for (int i = 0; i < 20 && (i == 0||s[i - 1] != '\0'); i++) {
				fs->directory[last_index].name[i] = s[i];
				name_count++;
			}
			fs->directory[last_index].sibling = NULL;
			fs->directory[last_index].child = NULL;
			fs->directory[last_index].parent = current_index;
			fs->directory[index_directory].sibling = last_index;
			fs->directory[last_index].identity = 0;
			fs->directory[last_index].size = 0;
			fs->directory[current_index].size += name_count;
			fs->directory[last_index].modified_date = gtime;
			fs->directory[last_index].create_date = gtime_create;
		}
		last_index++;
		printf("last index = %d\n", last_index);
		printf("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\after MKDIR\n");
		display(fs);
	}
	else if (op == 4) {
		//CD
		if (fs->directory[current_index].child == NULL) printf("no subdirectory error\n");
		int index_directory = fs->directory[current_index].child;
		while (fs->directory[index_directory].name != s && fs->directory[index_directory].sibling != NULL) index_directory = fs->directory[index_directory].sibling;
		if (fs->directory[index_directory].name != s) printf("no such directory error\n");
		else if (fs->directory[index_directory].identity == 1) printf("can not move into a file\n");
		else current_index = index_directory;
		current_depth++;
	}
	//RM_RF
	else if (op == 7) {
		int index_previous = -1;
		if (fs->directory[current_index].child == NULL) printf("no subdirectory error\n");
		int index_directory = fs->directory[current_index].child;
		while (fs->directory[index_directory].name != s && fs->directory[index_directory].sibling != NULL) {
			index_previous = index_directory;
			index_directory = fs->directory[index_directory].sibling;
		}
		if (fs->directory[index_directory].name != s) printf("no such directory error\n");
		//clean the file inside
		// it is file
		if (fs->directory[index_directory].identity == 1) fs_gsys(fs, 2, fs->directory[index_directory].name);
		// it is direcoty
		// clean the directory
		if (index_previous != -1) fs->directory[index_previous].sibling = NULL;
		else fs->directory[current_index].child = NULL;
	}
}
