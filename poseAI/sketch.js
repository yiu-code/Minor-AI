let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "";

let state = 'waiting';
let targetLabel;

let startLogging = false;

function keyPressed() {
  if (key == 't') {
	  setTimeout(function() {
		startLogging = true;
		setTimeout(function() {
			startLogging = false}, 
			3000)
	  }, 5000)
  }
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);

  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true,
    layers:[
      {
        type: 'dense',
        units: 64,
        activation: 'relu',
      },
      {
        type: 'dense',
        units: 16,
        activation: 'relu',
      },
      {
        type: 'dense',
        activation: 'softmax',
      },
    ] 
  }
  brain = ml5.neuralNetwork(options);
  
   //LOAD PRETRAINED MODEL
   //Uncomment to train your own model!
  /* const modelInfo = {
	model: 'yoga/yoga.json',
	metadata: 'yoga/yoga_meta.json',
	weights: 'yoga/yoga.weights.bin',
   };
  brain.load(modelInfo, brainLoaded); */

  // LOAD TRAINING DATA
  brain.loadData('train.json', dataReady);
}

function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) { 
  if (startLogging) {console.log(results);}
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
  } else {
	poseLabel = '';
  }
  else{
    poseLabel = ""
  }
  classifyPose();
}

function dataReady() {
  brain.normalizeData();
  brain.train({
    epochs: 50,
    batchsize: 24
  }, finished);
}

function finished() {
  console.log('model trained');
  //brain.save('yoga'); --UNCOMMENT WHEN YOU WANT TO SAVE THE MODEL
  classifyPose();
}

function gotPoses(poses) {
  // console.log(poses); 
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}

function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
  pop();

  fill(255, 0, 255);
  noStroke();
  textSize(50);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);
}