#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);

__visible struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0xdc02b68, __VMLINUX_SYMBOL_STR(module_layout) },
	{ 0x3f7aea1f, __VMLINUX_SYMBOL_STR(wake_up_process) },
	{ 0x1c7dfacc, __VMLINUX_SYMBOL_STR(kthread_create_on_node) },
	{ 0xeac5b58b, __VMLINUX_SYMBOL_STR(_do_fork) },
	{ 0x56aa7d, __VMLINUX_SYMBOL_STR(current_task) },
	{ 0x3df3c334, __VMLINUX_SYMBOL_STR(put_pid) },
	{ 0xf37409c9, __VMLINUX_SYMBOL_STR(do_wait) },
	{ 0x1fb67182, __VMLINUX_SYMBOL_STR(find_get_pid) },
	{ 0xdb7305a1, __VMLINUX_SYMBOL_STR(__stack_chk_fail) },
	{ 0x952664c5, __VMLINUX_SYMBOL_STR(do_exit) },
	{ 0xa750c43, __VMLINUX_SYMBOL_STR(do_execve) },
	{ 0xa1faca03, __VMLINUX_SYMBOL_STR(getname) },
	{ 0x50eedeb8, __VMLINUX_SYMBOL_STR(printk) },
	{ 0xb4390f9a, __VMLINUX_SYMBOL_STR(mcount) },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";


MODULE_INFO(srcversion, "7C2F1BFE44DB57FCB214CD9");
