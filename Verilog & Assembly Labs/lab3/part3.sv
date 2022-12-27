`timescale 1ns / 1ns // `timescale time_unit/time_precision
//module x(A, B, Function, ALUout);
//	part3 #(6) u6(A, B, Function, ALUout);
	
//endmodule
module part3(A, B, Function, ALUout);
	parameter N = 6; 
	input logic [N-1:0] A, B; 
	input logic [2:0] Function;
	output logic [2*N-1:0] ALUout;
	
	logic [N:0] add_out;
	logic [2*N-1:0] combined; 

	assign combined = {A, B};
	
	assign add_out = A + B; 

	always_comb
	begin
	case ( Function )	
	0: ALUout = {{(N-1)*{1'b0}}, add_out}; 
	1: if (|combined == 1)
		ALUout = 8'b00000001;
		else
		ALUout = 8'b00000000;
	
	2: if (&combined == 1)
		ALUout = 8'b00000001;
		else
		ALUout = 8'b00000000; 
		
	3: ALUout = combined;
	default: ALUout = 8'b00000000;

	endcase
	end

endmodule


module part1(a,b,c_in, s, c_out);
	input logic [3:0] a, b; 
	input logic c_in; 
	
	output logic [4:0] s;	
	output logic [3:0] c_out;
	
	FA u1(a[0], b[0], c_in, c_out[0], s[0]);
	FA u2(a[1], b[1], c_out[0],c_out[1], s[1]);
	FA u3(a[2], b[2], c_out[1], c_out[2], s[2]);
	FA u4(a[3], b[3], c_out[2], c_out[3], s[3]);
	
	assign s[4] = c_out[3];
	
	

endmodule
	
module FA(a,b,c_in, c_out, s);
	input logic a, b, c_in;
	output logic c_out, s; 
	
	assign c_out = c_in & a | c_in & b | b & a;
	assign s = a ^ b ^ c_in; 

endmodule
