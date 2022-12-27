.global _start
.text
_start:
la s2, LIST
addi s10, zero, 0 #sum 
addi s11, zero, 0 #counter 
# Write your code here
lw s3 0(s2)
addi s5, zero, -1
addi s6, zero, 0

loop: beq s3, s5 END

addi s2, s2, 4 #increment the base by 4 
add s10, s10, s3 #sum 
lw s3 0(s2) #load the word 

addi s11, s11, 1 #counter 

j loop

END:
ebreak
.global LIST
.data
LIST:
.word 1,2,3,5,0xA, -1
