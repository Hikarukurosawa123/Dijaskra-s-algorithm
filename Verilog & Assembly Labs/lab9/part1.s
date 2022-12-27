.global _start
.text
_start:

li s0, 0xffff0004 #where the ascii code of the key will be written 
li s1, 0xffff0000 #stores the enable signal (changes to 1 when the key is pressed) 
li s2, 0xffff0008 #check ready bit 
li s3, 0xffff000c #address to store the ASCII code (ready bit is set to 0 until another input is accepted)
li s7, 1 #for comparison 
#set delay length to 1 instruction

POLL:
lw s4 0(s1) #checks for the input enable 
beqz s4, POLL

lw s5 0(s0) #load the ascii code 

CHECK:
lw s6 0(s2) #load the enable bit value 
beq s6, s7, STORE

STORE: 
sw s5, 0(s3)
j POLL

END: 
ebreak 

