`timescale 1ns / 1ns // `timescale time_unit/time_precision

//SW[2:0] data inputs
//SW[9] select signals

//LEDR[0] output display

module hex_decoder(c,display);
	input logic [3:0] c; 
	//logic [6:0] h; 
	output logic [6:0] display; 

	
	/*
	assign display[0] = (~c[3] | ~c[2] | ~c[1] | c[0]) & (~c[3] | c[2] | ~c[1] | ~c[0]) & (c[3] | ~c[2] | c[1] | c[0])&(c[3] | c[2] | ~c[1] | c[0]);
	assign display[1] = (~c[3] | c[2] | ~c[1] | c[0]) & (~c[3] | c[2] | c[1] | ~c[0]) & (c[3] | ~c[2] | c[1] | c[0]) & (c[3] | c[2] | ~c[1] | ~c[0]) & (c[3] | c[2] | c[1] | ~c[0]) & (c[3] | c[2] | c[1] | c[0]);  
	assign display[2] = (~c[3] | ~c[2] | c[1] | ~c[0]) & (c[3] | c[2] | c[1] | ~c[0]) & (c[3] | c[2] | ~c[1] | ~c[0]) & (c[3] | c[2] | c[1] | c[0]);  
	assign display[3] = (~c[3] | ~c[2] | ~c[1] | c[0]) & (~c[3] | c[2] | ~c[1] | ~c[0]) & (~c[3] | c[2] | c[1] | c[0]) & (c[3] | ~c[2] | ~c[1] | c[0]) & (c[3] | ~c[2] | c[1] | ~c[0]) & (c[3] | c[2] | c[1] | c[0]);
	assign display[4] = (~c[3] | ~c[2] | ~c[1] | c[0]) & (~c[3] | ~c[2] | c[1] | c[0]) & (~c[3] | c[2] | ~c[1] | ~c[0]) & (~c[3] | c[2] | ~c[1] | c[0]) & (~c[3] | c[2] | c[1] | c[0])& (c[3] | ~c[2] | ~c[1] | c[0]); 
	assign display[5] = (~c[3] | ~c[2] | ~c[1] | c[0]) & (~c[3] | ~c[2] | c[1] | ~c[0]) & (~c[3] | ~c[2] | c[1] | c[0]) & (c[3] | c[2] | ~c[1] | c[0]); 
	assign display[6] = (~c[3] | ~c[2] | ~c[1] | ~c[0]) & (~c[3] | ~c[2] | ~c[1] | c[0]) & (~c[3] | c[2] | c[1] | c[0]) & (c[3] | c[2] | ~c[1] | ~c[0]); 
	//seg7 conversion 
	*/
	
	
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
