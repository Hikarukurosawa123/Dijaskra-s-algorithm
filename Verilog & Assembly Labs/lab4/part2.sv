`timescale 1ns / 1ns // `timescale time_unit/time_precision

module part2(Clock, Reset_b, Data, Function, ALUout);
	input logic Clock, Reset_b; 
	input logic [1:0] Function; 
	output logic [7:0] ALUout; 
	logic [3:0] B; 
	logic [7:0] out; 	
	input logic [3:0] Data;


	assign B = ALUout[3:0];	


	always_ff@(posedge Clock) // active high synchronous reset 
	begin 
	if(Reset_b)
		ALUout <= 8'b00000000;
	else 
		ALUout <= out;
	end 

	always_comb
	begin 
		case(Function)
		0: out = Data+B; 
		1: out = Data*B; 
		2: out = B<<Data; 
		default: out = ALUout;
		endcase 
	end 
endmodule
	