`timescale 1ns / 1ns // `timescale time_unit/time_precision

module part3(clock, reset, ParallelLoadn, RotateRight, ASRight, Data_IN, Q);
	input logic clock; 
	input logic reset; 
	input logic [3:0] Data_IN;
	input logic ParallelLoadn; 
	input logic RotateRight;
	input logic ASRight;
	output logic [3:0] Q;
	
	logic [3:0] out;
	logic temp;
	
	always_comb
		if(ParallelLoadn && RotateRight == 0)
		begin
		out[3] = Q[2];
		out[2] = Q[1];
		out[1] = Q[0];
		out[0] = Q[3];
		end
		else if(ParallelLoadn && RotateRight && ASRight == 0)
		begin
		out[2] = Q[3];
		out[1] = Q[2];
		out[0] = Q[1];
		out[3] = Q[0];
		end
		else if(ParallelLoadn && RotateRight && ASRight == 1)
		begin 
		out[3] = Q[3];
		out[2] = Q[3];
		out[1] = Q[2];
		out[0] = Q[1];
		end
		
	
	
		
	shift u0(Data_IN[0],out[0],  ParallelLoadn, clock, reset, Q[0]);
	shift u1(Data_IN[1], out[1],  ParallelLoadn, clock, reset, Q[1]);
	shift u2(Data_IN[2], out[2] , ParallelLoadn, clock, reset,  Q[2]);
	shift u3(Data_IN[3], out[3],ParallelLoadn, clock, reset,  Q[3]);




endmodule

module shift(Data_IN, out,ParallelLoadn, clock, reset, Q);
	input logic clock; 
	input logic reset; 
	input logic ParallelLoadn;
	input logic Data_IN;
	output logic Q;
	input logic out; 

	always_ff@(posedge clock)
	begin

	if(reset)
		Q <= 1'b0;
	else
		if(ParallelLoadn==0)
		Q <= Data_IN;
		else
		Q <= out;
		
		
	end
endmodule
		