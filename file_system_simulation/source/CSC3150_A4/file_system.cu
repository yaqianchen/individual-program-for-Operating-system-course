﻿#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 gtime_create = 0;
__device__ __managed__ u32 block_position = 0;
__device__ __managed__ u32 FCB_position = 4096;
__device__ __managed__ u32 current_FCB_position = 4096;

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

__device__ __managed__ struct FCB_node
{
	char name[20];
	u32 date;
	u32 size;
} node;

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
	char name[20];
	if (op == 0) {
		printf("===sort by modified time===\n");
		for (u32 i = 4096; i <= stop_position; i = i + 32) {
			for (int j = 4; j < 24 ; j++) {
				name[j - 4] = fs->volume[i + j];
			}
			printf("%s\n",name);
		}
	}
	else {
		u32 size;
		printf("===sort by file size===\n");
		for (u32 i = 4096; i <= stop_position; i = i + 32) {
			for (int j = 4; j < 24 ; j++) {
				name[j - 4] = fs->volume[i + j];
			}
			size = (fs->volume[i] << 24) + (fs->volume[i + 1] << 16) + (fs->volume[i + 2] << 8) + (fs->volume[i + 3]);
			printf("%s %d\n", name, size);
		}
	}
}

__device__ void swap(FileSystem* fs, u32 x, u32 y) {
	for (int i = 0; i < 32; i++) {
		uchar tempt = fs->volume[x + i];
		fs->volume[x + i] = fs->volume[y + i];
		fs->volume[y + i] = tempt;
	}
}

__device__ void bubblesort(FileSystem *fs, u32 left, u32 right, int op) {

	// sort by date
	if (op == 0) {
		for (int i = left; i < right; i = i + 32) {
			for (int j = left; j < right - i + left; j = j + 32) {
				u32 j_date_previous =  (fs->volume[j + 26] << 8) + (fs->volume[j + 27]);
				u32 j_date_after = (fs->volume[j + 26 + 32] << 8) + (fs->volume[j + 27 + 32]);
				if (j_date_previous < j_date_after) swap(fs, j, j + 32);
			}
		}
	}
	else {
		for (int i = left; i < right; i = i + 32) {
			for (int j = left; j < right - i + left; j = j + 32) {
				u32 j_size_previous = (fs->volume[j] << 24) + (fs->volume[j + 1] << 16) + (fs->volume[j + 2] << 8) + (fs->volume[j + 3]);
				u32 j_size_after = (fs->volume[j + 32] << 24) + (fs->volume[j + 1 + 32] << 16) + (fs->volume[j + 2 + 32] << 8) + (fs->volume[j + 3 + 32]);
				u32 j_date_previous = (fs->volume[j + 24] << 8) + (fs->volume[j + 25]);
				u32 j_date_after = (fs->volume[j + 24 + 32] << 8) + (fs->volume[j + 25 + 32]);
				if (j_size_previous < j_size_after) swap(fs, j, j + 32);
				if (j_size_after == j_size_previous && j_date_previous > j_date_after) swap(fs, j, j + 32);
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

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	//if not exist
	if (if_exist(fs, s) == -1) {
		if (op == 0) {
			printf("can not find the file to read error\n");
			return -1;
		}

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

	/* Implement read operation here */
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
    //enough space
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

__device__ void fs_gsys(FileSystem *fs, int op)
{
	u32 stop_point;

	/* Implement LS_D and LS_S operation here */
	for (u32 i = 4096; i - 32 < 36863; i = i + 32) {
		u32 size = (fs->volume[i] << 24) + (fs->volume[i + 1] << 16) + (fs->volume[i + 2] << 8) + (fs->volume[i + 3]);
		if (size == 0) {
			size = (fs->volume[4096] << 24) + (fs->volume[4096 + 1] << 16) + (fs->volume[4096 + 2] << 8) + (fs->volume[4096 + 3]);
			stop_point = i - 32;
			break;
		}
		stop_point = i - 32;
	}

	//if there is no file
	if (stop_point < 4096) printf("no file in FCB error\n");

	//sort by date
	if (op == 0) {
		bubblesort(fs, 4096, stop_point, 0);
		display(fs, stop_point, 0);
	}

	//sort by size
	else {
		bubblesort(fs, 4096, stop_point, 1);
		display(fs, stop_point, 1);
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
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
}
