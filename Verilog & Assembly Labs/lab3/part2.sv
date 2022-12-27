`timescale 1ns / 1ns // `timescale time_unit/time_precision
module part2(A, B, Function, ALUout);
	input logic [3:0] A, B; 
	input logic [2:0] Function;
	output logic [7:0] ALUout;
	
	logic [4:0] add_out;
	
	logic [7:0] combined; 
	logic and_state, or_state; 

	assign combined = {A, B};
	assign or_state = |combined;
	assign and_state = &combined;  
	
	part1 u5(A,B, 0, add_out, last); 

	always_comb
	begin
	case ( Function )	
	0: ALUout = {3'b000, add_out}; 
	1: if (or_state == 1)
		ALUout = 8'b00000001;
		else
		ALUout = 8'b00000000;
	
	2: if (and_state == 1)
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

module hex_decoder(c,display);
	input logic [3:0] c; 
	//logic [6:0] h; 
	output logic [6:0] display; 
	
	always_comb
	begin 
		case(c)
			0: display = 7'b1000000; 
			1: display = 7'b1111001; 
			2: display = 7'b0100100; 
			3: display = 7'b0110001; 
			4: display = 7'b0010001; 
			5: display = 7'b0010010; 
			6: display = 7'b0000010; 
			7: display = 7'b1011000; 
			8: display = 7'b0000000; 
			9: display = 7'b0011000; 
			10: display = 7'b0001000; 
			11: display = 7'b0000011; 
			12: display = 7'b1000110; 
			13: display = 7'b0100001; 
			14: display = 7'b0000110; 
			15: display = 7'b0001110; 
			default: display = 7'b0000000; 
		endcase
	end 
	
endmodule
