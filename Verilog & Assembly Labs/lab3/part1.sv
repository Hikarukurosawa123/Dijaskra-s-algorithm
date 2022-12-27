`timescale 1ns / 1ns // `timescale time_unit/time_precision



module part1(a,b,c_in, s, c_out);
	input logic [3:0] a, b; 
	input logic c_in; 
	
	output logic [3:0] s;	
	output logic [3:0] c_out;
	
	FA u1(a[0], b[0], c_in, c_out[0], s[0]);
	FA u2(a[1], b[1], c_out[0],c_out[1], s[1]);
	FA u3(a[2], b[2], c_out[1], c_out[2], s[2]);
	FA u4(a[3], b[3], c_out[2], c_out[3], s[3]);
	
	

endmodule
	
module FA(a,b,c_in, c_out, s);
	input logic a, b, c_in;
	output logic c_out, s; 
	
	assign c_out = c_in & a | c_in & b | b & a;
	assign s = a ^ b ^ c_in; 

endmodule