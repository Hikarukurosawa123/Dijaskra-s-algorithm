.data
# Messages
msg_1: .asciz "Please take a deep breath     \n"
msg_2: .asciz "Please drink some water       \n"
msg_3: .asciz "Please give your eyes a break \n"
# Timer Related
timeNow: .word 0xFFFF0018 # current time
cmp: .word 0xFFFF0020 # time for new interrupt
counter: .word 0
.text
# Display Related
.eqv OUT_CTRL 0xffff00084
.eqv OUT 0xffff000C

main:
la s0, msg_1 #first message 
addi sp, sp, -16
sw s0, 0(sp)
addi sp, sp, 16

li s3, 0xffff000c #address to store the ASCII code
# Set time to trigger interrupt to be 5000 milliseconds (5 seconds)
la s1, cmp #address to store the timer value 
lw s1, 0(s1) #value of the address to store the timer value 
li s2, 5000 #5 seconds 
sw s2 0(s1) #set timer
# Set the handler address and enable interrupts
la t0, timer_handler #address of the handler
csrrw zero, utvec, t0 #handler address 
csrrsi zero, ustatus, 0x1 #set interrupt enable bit 
csrrsi zero, uie, 0x10 #set timer interrupt enable 
addi t1, zero, 31
addi t2, zero, 0
addi t3, zero, 0
addi s7, zero, 0 
li t6, 5000
addi s6, zero, 0
addi s8, zero, 0
addi s9, zero, 0
# Loop over the messages
addi s9, zero, 31
addi s10, zero, 63
addi s11, zero, 95
addi s7, zero, 0 #keeps track of when to loop back

LOOP:
beq s6, t1, NEXT
addi s7, s7, 1
lb s5, 0(s0) 
sb s5, 0(s3)
addi s0, s0, 1
addi s6, s6, 1
j LOOP
#find a way to move to the next message 
#change the address of s0 to the next message
NEXT:
beq s6, s9, TO_MSG2
beq s6, s10, TO_MSG3
beq s6, s11, TO_MSG1
j WAIT

TO_MSG2:
addi s0, s0, 1
addi s6, s6, 1
addi t1, t1, 32
j WAIT
TO_MSG3:
addi s0, s0, 1
addi s6, s6, 1
addi t1, t1, 32
j WAIT
TO_MSG1:
addi sp, sp, -16
lw s0, 0(sp)
addi sp, sp, 16
addi s6, zero, 0
addi t1 zero, 31
j WAIT

WAIT:
beqz s7 LOOP
j WAIT

# Print message to ASCII display
timer_handler:
addi sp, sp, -8
sw t2, 0(sp)
sw t3, 4(sp)
# Push registers to the stack
# Indicate that 5 seconds have passed
la t3, timeNow #load the address of hte time NOW
lw t2, 0(t3) #get the value of address
lw t2, 0(t2)#get value of time 
add t2, t2, t6 #assign new timer value
addi s7, zero, 0 
sw t2, 0(s1)
# Pop registers from the stack

lw t2, 0(sp) #load it back
lw t3, 4(sp)
addi sp, sp, 8
uret
