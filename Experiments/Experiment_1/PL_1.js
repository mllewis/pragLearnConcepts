// ---------------- HELPER ------------------
// show slide function
function showSlide(id) {
  $(".slide").hide(); //jquery - all elements with class of slide - hide
  $("#"+id).show(); //jquery - element with given id - show
}

//array shuffle function
shuffle = function (o) { //v1.0
    for (var j, x, i = o.length; i; j = parseInt(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
    return o;
}
//--------------- STIMULUS SET-UP --------------

var filename = "PL_1x";
var condCounts = "1,100;2,100;3,100;4,100";
var xmlHttp = null;
xmlHttp = new XMLHttpRequest();
xmlHttp.open( "GET", "http://langcog.stanford.edu/cgi-bin/subject_equalizer/maker_getter.php?conds=" + 
		condCounts +"&filename=" + filename, false );
xmlHttp.send( null );
var cond = xmlHttp.responseText;

var testObjectsUnshuffled = ["00", "01", "10", "11"];
var testObjects = ["00", "01", "10", "11"];
var numFlowersinField = 16;
var questionmark = "images/question.jpg";

//both petals and thorns have a base rate of 0.25: 
var flowerField = ["00", "00", "00", "00", "00", "00", "00", "00", "00", "01", "01", "01", "10", "10", "10", "11"];
var counter1 = 0;
var counter2 = 1;

//begin by shuffling testObjects and field of flowers
shuffle(testObjects);
shuffle(flowerField);

//pick flower name
var name = shuffle(["wug", "fep", "dax", "bapi", "kiba", "tupa"])[0];

//--------- CONTROL FLOW ----------------
//PRELOAD ALL IMAGES
var allImageNames = ["question", "00", "01", "10", "11", "circled01", "circled10", "point01", "point10"];
var images = new Array() // By creating image object and setting source, images preload
for (i=0;i<allImageNames.length;i++) {
    images[i] = new Image()
    images[i].src = "images/" + allImageNames[i] + ".jpg";
} 

showSlide("instructions");

// MAIN EXPERIMENT
var experiment = {
	flowerName: name,
	condition: "noContext_Speaker",
	context: "context",
	speaker: "speaker",
	example: "",
	nofeature: 0,
	samefeature: 0,
	difffeature: 0,
	bothfeatures: 0,
	check_correct: "FALSE",
	about: "",
	comment: "",

	exposure: function() {
		showSlide("exposure");  
		
		var flowers_html = '<table id="critTable"><tr>';
		for (i=0; i<numFlowersinField;i++) {
			flowers_html += '<td style="text-align: center;"><img class="mysteryFlowers" src="' + questionmark + '" id="flower_' + i + '"></td>';		
			if ((i+1)%4 === 0) {
				flowers_html += "</tr><tr>";
			}
		}
		flowers_html += '</tr></table>';
		$("#field").html(flowers_html); 

		$('.mysteryFlowers').bind('click', function(event) {
			counter1++;
	   		var flowerID = $(event.currentTarget).attr('id');
	   		//flowers with id between 10 and 15
	   		if (flowerID.length > 8) {
				document.getElementById(flowerID).src = "images/" + flowerField[10 + parseInt(flowerID.charAt(8))] + ".jpg";
	   		} else {
	   			//flowers with id between 0 and 9
	    		document.getElementById(flowerID).src = "images/" + flowerField[parseInt(flowerID.charAt(7))] + ".jpg";
	    	}
  		});
	},

	teaching: function() {
		if (counter1 < numFlowersinField) {
			$("#error").html('<table cellspacing="2" align="center"><tr><td id="message"><font color="red">Please click on all the flowers to continue.</font></td></tr></table>');
			return;
		}

		//set condition variable
		if (cond == 2) {
			experiment.condition = "Context_Speaker";
		} else if (cond == 3) {
			counter2 = 0;
			experiment.condition = "noContext_noSpeaker";
			experiment.context = "no context";
			experiment.speaker = "no speaker";
		} else if (cond == 4) {
			counter2 = 0;
			experiment.condition = "Context_noSpeaker";
			experiment.speaker = "no speaker";
		}

		showSlide("stage");
		var instructions_html = "";
		var example_html = '<table id="critTable"><tr>';;

		//take either 10 or 01 to be example (chosen at random)
		experiment.example = ((shuffle(["01", "10"])).splice(0,1))[0];

		//for condition 4: scramble example with 11 context, and add a circle to the example wug
		var wug = "circled" + experiment.example;
		var context = shuffle(["11", wug]);

		//for condition 2: scramble example with 11 context, and add a point to the example wug
		if (cond == 2) {
			wug = "point" + experiment.example;
			context = shuffle(["11", wug]);
		}

		if (cond == 1) {
			instructions_html += "<p class='block-text'>Your friend Sally shows you this flower and tells you that it is a <b>" + experiment.flowerName + "</b>.</p>"
			example_html += '<td style="text-align: center;"><img  src="images/'+ experiment.example +'.jpg" class="objImage"/></td></tr></table>'
		}
		if (cond == 2) {
			instructions_html += "<p class='block-text'>Your friend Sally shows you these two flowers and tells you that the one she points to is a <b>" + experiment.flowerName + "</b>.</p>"
			for (i=0;i<2;i++){
		  		example_html += '<td style="text-align: center;"><img  src="images/'+ context[i] +'.jpg" class="objImage"/></td>';
		  	}
		  	example_html += '</tr></table>';
		}
		if (cond == 3) {
			instructions_html += "<p class='block-text'>Here is a new field of flowers. Click on one of the question marks to learn the name of one of the flowers.</p>"
			for (i=0; i<numFlowersinField;i++) {
				example_html += '<td style="text-align: center;"><img class="choices" src="' + questionmark + '" id="choice_' + i + '"></td>';		
				if ((i+1)%4 === 0) {
					example_html += "</tr><tr>";
				}
			}
			example_html += '</tr></table>';
		}
		if (cond == 4) {
			instructions_html += "<p class='block-text'>On a hike, you run into two flowers. Click on a question mark to pick a flower and learn its name.</p>";
			for (i=0; i<2;i++) {
				example_html += '<td style="text-align: center;"><img class="choices" src="' + questionmark + '" id="choice_' + i + '"></td>';		
			}
			example_html += '</tr></table>';
		}

		$("#labelInst").html(instructions_html);
		$("#example").html(example_html);



		if (cond == 3) {
			var called = false;
			$('.choices').one('click', function(event) {
				if (called === false) {
					called = true;
					counter2++;
	   				var choiceID = $(event.currentTarget).attr('id');
	   				document.getElementById(choiceID).src = "images/" + experiment.example + ".jpg";
	   				var name_html = "<p class='block-text'>You picked a <b>" + experiment.flowerName + "</b>.</p>";
	   				$("#name").html(name_html);
	   			}
	   		});
	   	}	

		if (cond == 4) {
			var called = false;
			$('.choices').one('click', function(event) {
				if (called === false) {
					called = true;
					counter2++
		   			var choiceID = $(event.currentTarget).attr('id');

		   			//LEFT click
		   			if (choiceID === "choice_0") {
		   				document.getElementById(choiceID).src = "images/" + wug + ".jpg";
		   				document.getElementById("choice_1").src = "images/" + "11" + ".jpg";

		   			//RIGHT click
		   			} else {
						document.getElementById(choiceID).src = "images/" + wug + ".jpg";
		   				document.getElementById("choice_0").src = "images/" + "11" + ".jpg";
		   			}
		   			var name_html = "<p class='block-text'>You picked a <b>" + experiment.flowerName + "</b>.</p>";
		   			$("#name").html(name_html);
		   		}
	   		});
	   	}
	},

	query: function() {
		//catch for cond3 and 4
		if (counter2 < 1) {
			return;
		}
	 
	 	showSlide("query");

	  	var finalinst_html = "<p class='block-text'>Here are 4 new flowers. Which of them are <b>" + experiment.flowerName + "s</b>? Place a bet on each flower from 0 to 100, where 0 means that you are certain the flower is <b>not</b> a " + experiment.flowerName + " and 100 means that you are certain it is a " + experiment.flowerName + ".</p>";
	  	$("#finalinstructions").html(finalinst_html);

	    // Create the object table (tr=table row; td= table data)
	    var objects_html = '<table id="critTable"><tr>';
	    var name;
		for (i=0;i<4;i++){
		  name = testObjects[i] + ".jpg";
		  objects_html += '<td style="text-align: center;"><img  src="images/'+ name +'" class="objImage"/></td>';
		}
		objects_html += '</tr><tr>';

		for (i=0;i<4;i++) {
		  objects_html += '<td style="text-align: center;"><input type="text" id="item_'+i+'"  /><br></td>';
		}

		objects_html += '</tr></table>';	
		$("#allTestObjects").html(objects_html) //jquery - $find the object in the DOM with the id of object, 
	},

    finished: function() {
    	if ((!isNaN(document.getElementById('item_0').value) && parseInt(document.getElementById('item_0').value) >= 0 && parseInt(document.getElementById('item_0').value) <= 100) &&
    		(!isNaN(document.getElementById('item_1').value) && parseInt(document.getElementById('item_1').value) >= 0 && parseInt(document.getElementById('item_1').value) <= 100) &&
    		(!isNaN(document.getElementById('item_2').value) && parseInt(document.getElementById('item_2').value) >= 0 && parseInt(document.getElementById('item_2').value) <= 100) &&
    		(!isNaN(document.getElementById('item_3').value) && parseInt(document.getElementById('item_3').value) >= 0 && parseInt(document.getElementById('item_3').value) <= 100)) {

	  		//save results
    		experiment.nofeature = parseInt(document.getElementById('item_' + testObjects.indexOf("00")).value);
    		experiment.bothfeatures = parseInt(document.getElementById('item_' + testObjects.indexOf("11")).value);
    		experiment.samefeature = parseInt(document.getElementById('item_' + testObjects.indexOf(experiment.example)).value);
		
    		if (experiment.example === "01") {
    			experiment.difffeature = parseInt(document.getElementById('item_' + testObjects.indexOf("10")).value);
    		} else {
    			experiment.difffeature = parseInt(document.getElementById('item_' + testObjects.indexOf("01")).value);
    		}

    		if (experiment.example === "01") {
    			experiment.example = "thorns"
    		} else {
    			experiment.example = "petals"
    		}

			showSlide("check");
			
			// Question 1
			var q1_html = "<p class='block-text'>(1) Which flower were you told was a <b>" + experiment.flowerName + "</b>?";
			$("#q1").html(q1_html);
	    	var objects_html = '<table id="critTable"><tr>';
	    	var name;
			for (i=0;i<4;i++){
			  	name = testObjectsUnshuffled[i] + ".jpg";
			  	objects_html += '<td style="text-align: center;"><img  src="images/'+ name +'" class="objImage"/></td>';
			}
			objects_html += '</tr><tr>';

			for (i=0;i<4;i++) {
		 		objects_html += '<td style="text-align: center;"><input type="radio" name="q1" id="radio_'+i+'"  /><br></td>';
			}
			$("#checkObjects").html(objects_html);

    	} else {
    		//error message if problem with bets
    		var message_html = '<table cellspacing="2" align="center"><tr><td id="message"><font color="red">Please make sure you have bet on every flower with a number from 0 to 100.</font></td></tr></table>';
			$("#message").html(message_html);
    	}
    },

    check_finished: function() {
		if (($("input[type=radio]:checked").length < 1) ||
			document.getElementById('about').value.length < 1) {
			$("#checkMessage").html('<font color="red">Please make sure you have answered all the questions!</font>');
		} else {
			//the more frequent object will always be presented on the right
			if (experiment.example === 'thorns' && document.getElementById('radio_1').checked ||
				experiment.example === 'petals' && document.getElementById('radio_2').checked) {
				experiment.check_correct = "TRUE";
			}
			experiment.about = document.getElementById("about").value;
			experiment.comment = document.getElementById("comments").value;

			experiment.end();
		}
    },

    end: function () {  
    	showSlide("done");
        setTimeout(function () {
			// Decrement only if this is an actual turk worker!		
			if (turk.workerId.length > 0){
				var xmlHttp = null;
				xmlHttp = new XMLHttpRequest();
				xmlHttp.open('GET',			 
						 'https://langcog.stanford.edu/cgi-bin/' + 
						 'subject_equalizer/decrementer.php?filename=' + 
						 filename + "&to_decrement=" + cond, false);
				xmlHttp.send(null);
			}

            turk.submit(experiment);
        }, 500); 
    }
}