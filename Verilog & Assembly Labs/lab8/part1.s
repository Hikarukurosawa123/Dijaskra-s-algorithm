.global _start
.text
_start:

addi a0,zero, 0 #set a0 
la a2, LIST # Load the memory address into a2
addi a4, zero, -1 #register used to check termination 
addi a1, zero, 0  #largest number of 1s at that moment 
addi s10, zero, 0 #return largest number of ones 
addi s2, zero, 0 #will be used for masking 
addi s1, zero, 0 

jal ONES
#return here
addi a2, a2, 4 #move to the next word  
lw s3, 0(a2) #load the word from the list 
beq s3, a4 END #terminate if the word is -1

ONES: #sub routine 
#run thorugh each word in the outside loop and check the numbef of ones for each word by running the subroutine 
#only use jal because the ra value would default back to 0) 
addi sp, sp, -12
sw s1, 0(sp)
sw s2, 4(sp)
sw s3, 8(sp)

LOOP:
beqz s3, ENDLOOP # Loop until data contains no more 1sS
srli s2, s3, 1 # Perform SHIFT, followed by AND
and s3, s3, s2
addi s1, s1, 1 # Count the numbver of ones so far
b LOOP

ENDLOOP: 
addi a0,zero, 0 #reset a0 value 
add a0, a0, s1 #return the value 
lw s1, 0(sp)
lw s2, 4(sp)
lw s3, 8(sp)
addi sp, sp, 12
bge a0, a1, STORE #checks if a0 is greater than a1
jr ra

STORE:
addi a1, a0, 0 #store the largest value into a1 

jr ra 

END:
add s10, s10, a1
ebreak

.global LIST
.data
LIST:
.word 85 3 5 12 0x103fe00f 0x103fe00f 0x103de00f 0x103ae00f 0x104fe00f 0x106fe00f -1
