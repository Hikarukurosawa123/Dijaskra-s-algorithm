.global _start
.text
_start:

la a0, LIST # Load the memory address into a0
######
lw s0, 0(a0)#number of elements 
addi sp, sp, -4
sw a0, 0(sp)
addi sp, sp, 4
addi a0, a0, 4
addi t1, zero, 0 #counter 
lw s1, 4(a0)#first element 
lw s2, 8(a0) #second element 

jal SWAP 
#outer for loop 
addi sp, sp, -4
lw a0, 0(sp)

addi a0, a0, 4
sw a0, 0(sp)
addi sp, sp, 4

lw s1, 0(a0)#first element 
lw s2, 4(a0) #second element 
bne t1, s0, SWAP #while counter not equal to the number of elements, do swap 
beqz s0 END #if the outside loop counter becomes 0, terminate 

addi sp, sp, -4
la a0, LIST # reset address of a0 to the first one 
sw a0, 0(sp)
addi sp, sp, 4
addi s0, s0, -1
addi t1, zero, 0
jr ra 

SWAP: # subroutine 
addi t1, t1, 1
bge s1, s2, CHANGE
addi a0, zero, 0 #return zero 
jr ra 

CHANGE:
addi t0, s1, 0
sw s2 0(a0) 
sw t0, 4(a0)
 
addi a0, zero, 1 #return one 
jr ra

END: 
ebreak



.global LIST
.data
LIST:
.word 10, 1400, 45, 23, 5, 3, 8, 17, 4, 20, 33
