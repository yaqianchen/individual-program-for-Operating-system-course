#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ int find_frame_number_in_frame_table(VirtualMemory *vm, u32 frame_number) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == frame_number) return i;
	}
	printf("out of index\n");
	return -1;
}

__device__ void change_frame_table_valid_to_invalid(VirtualMemory *vm, u32 frame_number) {
	int tempt = vm->invert_page_table[vm->PAGE_ENTRIES + find_frame_number_in_frame_table(vm, frame_number)];
	for (int i = find_frame_number_in_frame_table(vm,frame_number); i < vm->PAGE_ENTRIES - 1; i++) {
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = vm->invert_page_table[i + vm->PAGE_ENTRIES + 1];
	}
	vm->invert_page_table[2 * vm->PAGE_ENTRIES - 1] = tempt;
}

__device__ int find_frame_number(VirtualMemory *vm,u32 page_number) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_number) return i;
	}
	printf("out of index\n");
	return -1;
}

__device__ void move_to_storage(VirtualMemory *vm, u32 frame_number) {
	u32 page_number = vm->invert_page_table[frame_number];
	for (int i = 0; i < 32; i++) {
		vm->storage[page_number * 32 + i] = vm->buffer[frame_number * 32 + i];
	}
}

__device__ void memory_move_to_result(VirtualMemory *vm, uchar* result, u32 page_number) {
	u32 frame_number = find_frame_number(vm, page_number);
	for (int i = 0; i < 32; i++) {
		result[page_number * 32 + i] = vm->buffer[frame_number * 32 + i];
	}
}

__device__ void move_to_memory(VirtualMemory *vm, u32 frame_number, u32 page_number) {
	u32 original_page_number = vm->invert_page_table[frame_number];
	for (int i = 0; i < 32; i++) {
		vm->storage[original_page_number * 32 + i] = vm->buffer[frame_number * 32 + i];
		vm->buffer[frame_number * 32 + i] = vm->storage[page_number * 32 + i];
	}
	vm->invert_page_table[frame_number] = page_number;
}

__device__ bool check_page_fault(VirtualMemory *vm, u32 page_number) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_number) {
			return true;
		}
	}
	*vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
	return false;
}

__device__ void check_frame_full(VirtualMemory *vm, u32 page_number, u32 frame_number) {
	if (vm->invert_page_table[frame_number] != 0x80000000) {
		move_to_storage(vm, frame_number);
	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
	u32 offset = addr % 32;
	u32 page_number = addr / 32;
	u32 frame_number;
	if (!check_page_fault(vm,page_number)) {
		frame_number = vm->invert_page_table[vm->PAGE_ENTRIES];
		move_to_memory(vm, frame_number, page_number);
	}
	else {
		frame_number = find_frame_number(vm, page_number);
	}
	change_frame_table_valid_to_invalid(vm, frame_number);
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	/* Complete vm_write function to write value into data buffer */
	u32 offset = addr % 32;
	u32 page_number = addr / 32;
	u32 frame_number;
	if (!check_page_fault(vm, page_number)) {
		frame_number = vm->invert_page_table[vm->PAGE_ENTRIES];
		check_frame_full(vm, page_number, frame_number);
		vm->invert_page_table[frame_number] = page_number;
	}
	else {
		frame_number = find_frame_number(vm, page_number);
	}
	vm->buffer[frame_number * 32 + offset] = value;
	change_frame_table_valid_to_invalid(vm, frame_number);
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
	int input_size) {

	for (int i = 0; i < input_size; i++) {
		u32 page_number = i / 32;
		u32 frame_offset = i % 32;
		u32 frame_number;
		if (!check_page_fault(vm, page_number)) {
			frame_number = vm->invert_page_table[vm->PAGE_ENTRIES];
			move_to_memory(vm, frame_number, page_number);
		}
		else {
			frame_number = find_frame_number(vm, page_number);
		}
		memory_move_to_result(vm, results, page_number);
		change_frame_table_valid_to_invalid(vm, frame_number);
	} 
}

