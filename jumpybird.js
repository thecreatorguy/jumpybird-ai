"use strict";

//////////////////////
// Global Constants //
//////////////////////

// Update speed constants
const FPS          = 50;
const GAME_STEP_MS = 1000 / FPS;
const GAME_STEP_S  = 1 / FPS;

// Flappy bird graphic dimensions
const BIRD_WIDTH   = 30;
const COL_DISTANCE = 163;
const COL_WIDTH    = 40;
const FRAME_HEIGHT = 400;
const FRAME_WIDTH  = 550;
const HOLE_HEIGHT  = 90;

// Flappy bird game constants
const GRAVITY        = 1000; // pixels/s^2
const JUMP_THRESHOLD = 25;   // pixels between jumps
const MAX_SCORE      = 1000000;
const MOVE_SPEED     = 100;  // horizontal pixels/s
const TERMINAL_VEL   = 500;  // maximum vertical velocity

// Neural network and training constants
const DEFAULT_NEURONS_PER_LAYER = 8;
const MAX_GENERATIONS           = 1000000;
const MAX_LAYERS                = 10;
const MAX_NEURONS_PER_LAYER     = 50;
const MAX_POPULATION_SIZE       = 5000;

const RENDER_ON_SCORE_REACHED = [10, 100, 1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000];


//////////////////////
// Helper Functions //
//////////////////////

/**
 * Calculate the dot product of a 2d vector and a 1d vector
 *
 * Parameters:
 *  v2d: 2d vector, first ordered argument of dot product
 *  v1d: 1d vector, second ordered argument of dot product
 *
 * Return: 1d vector result of dot product
 */
function vecDot(v2d, v1d) {
    let res = [];
    for (let vec of v2d) {
        let val = 0;
        for (let i = 0; i < vec.length; i++) {
            val += vec[i] * v1d[i];
        }
        res.push(val);
    }
    return res;
}

/**
 * Calculate the vector addition of 2 1d vectors
 *
 * Parameters:
 *  v2d: 1d vector, first argument of vector addition
 *  v1d: 1d vector, second argument of vector addition
 *
 * Return: 1d vector result of vector addition
 */
function vecAdd(v1, v2) {
    let res = [];
    for (let i = 0; i < v1.length; i++) {
        res.push(v1[i] + v2[i]);
    }
    return res;
}

/**
 * The sigmoid activation function
 *
 * Parameters:
 *  val: real value to generate the sigmoid value of
 *
 * Return: result of sigmoid function on val
 */
function sigmoid(val) {
    return 1 / (1 + Math.exp(-val));
}

/**
 * Get a list of indices of an array (of arrays). Each element of the return value is an
 * ordered list representing the index of each value in the original array.
 * Ex: [[0, 1],  would produce the list: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2]]
 *      [2, 3],
 *      [4, 5, 6]]
 *
 * Parameters:
 *  arr: an array of any dimension to get indices of
 *
 * Return: array of indices
 */
function indicesOf(arr) {
    if (!arr.some(Array.isArray)) {
        return [...Array(arr.length).keys()];
    }

    let indices = [];
    for (let i = 0; i < arr.length; i++) {
        indices = indices.concat(indicesOf(arr[i]).map(subIndices => [i].concat(subIndices)));
    }

    return indices;
}

/**
 * Get a random int with the given maximum value, inclusive. If two arguments are given, then
 * gets a random integer within the range min-max, inclusive.
 *
 * Parameters:
 *  min: if one argument given, the maximum value of the range. If two are given, the minimum
 *  max: the maximum value of the range
 *
 * Return: random integer in the given range
 */
function randInt(min, max = null) {
    if (max === null) {
        max = min;
        min = 0;
    }
    max++;
    return Math.floor(Math.random() * max) + min;
}

/**
 * Gets a random element of the given array
 *
 * Parameters:
 *  arr: array to randomly choose from
 *
 * Return: random element of the given array
 */
function randChoice(arr) {
    return arr[randInt(arr.length - 1)];
}

/**
 * Function that will take an element's value and clamp it based on its min and max values
 *
 * Parameters:
 *  elem: number input to clamp
 */
function clampNumberInput(elem) {
    if (elem.valueAsNumber < elem.min) {
        elem.value = elem.min;
    }
    if (elem.valueAsNumber > elem.max) {
        elem.value = elem.max;
    }
}

/**
 * Adds an input and accompanying DOM elements to the given container
 *
 * Parameters:
 *  container: container DOM element to add the new elements to
 *  opts: object containing the following options-
 *    - id:    id of input
 *    - type:  type of input, range or number
 *    - min:   min value of input
 *    - max:   max value of input
 *    - step:  step value of input (range)
 *    - initVal: initial value of input
 *    - setVal: function that is passed the new value on input change
 */
function addInput(container, opts) {
    let label = document.createElement('label');
    label.setAttribute('for', opts.id);
    label.innerHTML = opts.title;
    container.append(label);

    let input = document.createElement('input');
    input.id = opts.id;
    input.setAttribute('type', opts.type);
    input.setAttribute('min', opts.min);
    input.setAttribute('max', opts.max);

    if (opts.type == 'range') {
        input.setAttribute('step', opts.step);
        let rangeAndDisplay = document.createElement('div');
        rangeAndDisplay.classList.add('range-and-display');
        rangeAndDisplay.append(input);

        let display = document.createElement('div');
        display.innerHTML = opts.initVal;
        input.addEventListener('input', function(e) {
            display.innerHTML = e.target.valueAsNumber;
            opts.setVal(e.target.valueAsNumber);
        });
        rangeAndDisplay.append(display);

        container.append(rangeAndDisplay);
    } else {
        input.addEventListener('change', function(e) {
            clampNumberInput(e.target);
            opts.setVal(e.target.valueAsNumber);
        });
        container.append(input);
    }
    input.value = opts.initVal; // must be set last to update range sliders visually
}

//////////////////////////////////////////
// Generic Genetic Neural Network Class //
//////////////////////////////////////////

/**
 * A generic neural network class that maps input label-value pairs to output-label pairs
 */
class NNetwork {

    /**
     * Generate a new neural network with the weights and biases randomized
     *
     * Parameters:
     *  inputLabels:  array of labels of the object that will be passed to the
     *                getOutputs() function
     *  outputLabels: array of labels that will be returned from the getOutputs() function
     *  hiddenShape:  array of lengths for the hidden layers. Each entry represents a hidden layer,
     *                the value of that entry is the number of neurons in that layer
     *  maxAbsVal:    maximum and minimum value for each weight and bias to be randomized to
     *
     * Returns: Randomized neural network with the given sizes
     */
    static newRandom(inputLabels, outputLabels, hiddenShape, maxAbsVal) {
        let lengths = [inputLabels.length, ...hiddenShape, outputLabels.length];

        let weights = [];
        for (let i = 1; i < lengths.length; i++) {
            let layerWs = []
            for (let j = 0; j < lengths[i]; j++) {
                let ws = [];
                for (let k = 0; k < lengths[i-1]; k++) {
                    ws.push(Math.random() * maxAbsVal * 2 - maxAbsVal);
                }
                layerWs.push(ws);
            }
            weights.push(layerWs);
        }

        let biases = [];
        for (let i = 1; i < lengths.length; i++) {
            let layer_biases = [];
            for (let j = 0; j < lengths[i]; j++) {
                layer_biases.push(Math.random() * maxAbsVal * 2 - maxAbsVal);
            }
            biases.push(layer_biases);
        }

        return new NNetwork(inputLabels, outputLabels, lengths, weights, biases);
    }

    /**
     * Duplicates a given neural network
     *
     * Parameters:
     *  net: the neural network to duplicate
     *
     * Return: duplicated neural network
     */
    static duplicate(net) {
        return new NNetwork(
            net.inputLabels,
            net.outputLabels,
            net.lengths,
            JSON.parse(JSON.stringify(net.weights)),
            JSON.parse(JSON.stringify(net.biases))
        );
    }

    /**
     * Generates the genetic offspring of two neural networks
     *
     * Parameters:
     *  parent1:         one neural network to generate offspring with
     *  parent2:         another neural network to combine with the first
     *  maxMutateChange: the maximum change to any weight or bias in the child networks
     *  mutateChance:    chance that any weight or bias will be mutated
     *
     * Return: array of two child neural networks
     */
    static generateOffspring(parent1, parent2, maxMutateChange, mutateChance) {
        let newNetworks = parent1._crossover(parent2);
        newNetworks.forEach(net => net._mutate(maxMutateChange, mutateChance));

        return newNetworks;
    }

    /**
     * !!private!! Creates a neural network out of the given data
     *
     * Parameters:
     *  inputLabels:  array of labels of the object that will be passed to the
     *                getOutputs() function
     *  outputLabels: array of labels that will be returned from the getOutputs() function
     *  lengths:      length of each layer, including input and output layers
     *  weights:      3d array of weights. The first dimension is the layer to multiply,
     *                the second is the neuron within the layer to find, and the last is the
     *                weight values to multiply the previous layers by.
     *  biases:       2d array of biases. The first dimension is the layer to add to,
     *                the second is the bias of the neuron within that layer
     */
    constructor(inputLabels, outputLabels, lengths, weights, biases) {
        this.inputLabels = inputLabels;
        this.outputLabels = outputLabels;
        this.lengths = lengths;
        this.weights = weights;
        this.biases = biases;
    }

    /**
     * Generate output pairs from the given input pairs by propagating the inputs through the network
     *
     * Parameters:
     *  inputs: object with at least the properties that correspond to the input labels
     *
     * Return: object with the properties that correspond to the output labels
     */
    getOutputs(inputs) {
        // Get the values from the inputs object and propagate them through the net
        let values = this.inputLabels.map(label => inputs[label]);
        for (let i = 0; i < this.weights.length; i++) {
            values = vecAdd(vecDot(this.weights[i], values), this.biases[i]).map(sigmoid);
        }

        // Because new generations will map to the same ordered outputs, we can arbitrarily
        // choose the output neurons to be mapped to the same ordered output labels
        let outputs = {};
        for (let i = 0; i < values.length; i++) {
            outputs[this.outputLabels[i]] = values[i];
        }

        return outputs;
    }

    /**
     * The 'crossover' step of genetic algorithms, combining this network with the other network by
     * randomly choosing which genes (weight or bias value) that will be swapped between the two
     *
     * Parameters:
     *  other: other neural network to perform the crossover step with
     *
     * Return: array of 2 new child neural networks
     */
    _crossover(other) {
        // Get new weights by splitting randomly between the two networks
        let weightIndices = indicesOf(this.weights);
        let numSplit = randInt(Math.round(0.2 * weightIndices.length), Math.round(0.8 * weightIndices.length));
        let splitIndices = [];
        for (let i = 0; i < numSplit; i++) {
            let choice = weightIndices.splice(randInt(weightIndices.length - 1), 1)[0];
            splitIndices.push({net: 0, layer: choice[0], col: choice[1], row: choice[2]});
        }
        splitIndices = splitIndices.concat(weightIndices.map(function(index) {
            return {net: 1, layer: index[0], col: index[1], row: index[2]};
        }));

        let net1Weights = JSON.parse(JSON.stringify(this.weights));
        let net2Weights = JSON.parse(JSON.stringify(this.weights));
        let netWeights = [this.weights, other.weights];
        for (let i of splitIndices) {
            net1Weights[i.layer][i.col][i.row] = netWeights[i.net][i.layer][i.col][i.row];
            net2Weights[i.layer][i.col][i.row] = netWeights[Math.abs(i.net - 1)][i.layer][i.col][i.row];
        }

        // Get new biases by splitting randomly between the two networks
        let biasIndices = indicesOf(this.weights);
        numSplit = randInt(Math.round(0.2 * biasIndices.length), Math.round(0.8 * biasIndices.length));
        splitIndices = [];
        for (let i = 0; i < numSplit; i++) {
            let choice = biasIndices.splice(randInt(biasIndices.length - 1), 1)[0];
            splitIndices.push({net: 0, layer: choice[0], neuron: choice[1]});
        }
        splitIndices = splitIndices.concat(biasIndices.map(function(index) {
            return {net: 1, layer: index[0], neuron: index[1]};
        }));

        let net1Biases = JSON.parse(JSON.stringify(this.biases));
        let net2Biases = JSON.parse(JSON.stringify(this.biases));
        let netBiases = [this.biases, other.biases];
        for (let i of splitIndices) {
            net1Biases[i.layer][i.neuron] = netBiases[i.net][i.layer][i.neuron];
            net2Biases[i.layer][i.neuron] = netBiases[Math.abs(i.net - 1)][i.layer][i.neuron];
        }

        // Return the two new networks
        let net1 = new NNetwork(this.inputLabels, this.outputLabels, this.lengths, net1Weights, net1Biases);
        let net2 = new NNetwork(this.inputLabels, this.outputLabels, this.lengths, net2Weights, net2Biases);

        return [net1, net2];
    }

    /**
     * Mutate this neural network by randomly changing weights and biases
     *
     * Parameters:
     *  maxChange:    the maximum change to any weight or bias
     *  mutateChance: chance that any weight or bias will be mutated
     */
    _mutate(maxChange, mutateChance) {
        let mutateVal = val => (Math.random() < mutateChance) ? val + Math.random() * maxChange * 2 - maxChange : val;
        this.weights = this.weights.map(
            layer => layer.map(
                nodeWeights => nodeWeights.map(mutateVal)));

        this.biases = this.biases.map(layer => layer.map(mutateVal));
    }
}

//////////////////////////////
// Flappy Bird Game Classes //
//////////////////////////////

/**
 * A bird to run through the gauntlet of the flappy game
 */
class Bird {
    /**
     * Generate a new random bird
     */
    static newRandom(inputs, hiddenShape) {
        return new Bird(NNetwork.newRandom(inputs, ['jump'], hiddenShape, 1));
    }

    /**
     * Duplicates a given bird
     *
     * Parameters:
     *  bird: the bird to duplicate
     *
     * Return: duplicated bird
     */
    static duplicate(bird) {
        return new Bird(NNetwork.duplicate(bird.controller));
    }

    /**
     * Generates the genetic offspring of two birds
     *
     * Parameters:
     *  parent1:         one neural network to generate offspring with
     *  parent2:         another neural network to combine with the first
     *  maxMutateChange: the maximum change to any weight or bias in the child networks
     *  mutateChance:    chance that any weight or bias will be mutated
     *
     * Return: array of two child birds
     */
    static generateOffspring(parent1, parent2, maxMutateChange, mutateChance) {
        let newNets = NNetwork.generateOffspring(parent1.controller,
            parent2.controller,
            maxMutateChange,
            mutateChance
        );
        let newBird1 = new Bird(newNets[0]);
        let newBird2 = new Bird(newNets[1]);

        return [newBird1, newBird2];
    }

    /**
     * Generates the next generation of birds from the old generation, choosing the best 10% of birds
     * to generate offspring from randomly, and mutating them.
     *
     * Parameters:
     *  oldGen:          old generation of birds, an array of birds that have finished running through a game
     *  maxMutateChange: largest change that the bird controller will make to its weights and biases
     *  mutateChance:    chance that any weight and bias in the controller network will be mutated
     *
     * Return: Array of birds based on the old generation, with the same number of birds as the old generation
     */
    static nextGen(oldGen, maxMutateChange, mutateChance) {
        let parents = oldGen.sort((a, b) => b.score - a.score)
            .slice(0, Math.round(0.1 * oldGen.length));

        let newBirds = parents.map(parent => Bird.duplicate(parent));
        if (parents[0].score < 100000) {
            let type = parents[0].controller;
            for (let i = 0; i < Math.round(0.1 * oldGen.length); i++) {
                newBirds.push(Bird.newRandom(type.inputLabels, type.lengths.slice(1, type.lengths.length - 1)));
            }
            while (newBirds.length < oldGen.length) {
                newBirds = newBirds.concat(
                    Bird.generateOffspring(randChoice(parents), randChoice(parents), maxMutateChange, mutateChance)
                );
            }
        } else {
            while (newBirds.length < oldGen.length) {
                let newBird = Bird.duplicate(randChoice(parents));
                newBird._mutate(maxMutateChange, mutateChance);
                newBirds = newBirds.concat(newBird);
            }
        }


        return newBirds.slice(0, oldGen.length);
    }

    /**
     * !!private!! Creates a new bird with the given controller neural network
     *
     * Parameters:
     *  contoller: neural network mapping any of the following to a jump output: x distance from the next hole,
     *             y distance from the next hole, y distance from the next next hole, current velocity
     */
    constructor(controller) {
        this.game = null;
        this.y = (FRAME_HEIGHT - BIRD_WIDTH) / 2;
        this.x = 0;
        this.velocity = 0;
        this.dead = false;
        this.lastJump = -500;
        this.score = 0;
        this.controller = controller;
    }

    _mutate(maxMutateChange, mutateChance) {
        this.controller._mutate(maxMutateChange, mutateChance);
    }

    /**
     * Finds if the bird should jump right now, given the current game state
     *
     * Return: boolean for if the bird should jump
     */
    jumps() {
        let nextCol = {x: FRAME_WIDTH, holeY: 0},
            nextCol2 = {x: FRAME_WIDTH, holeY: 0};
        for (let i = 0; i < this.game.cols.length; i++) {
            if (this.game.cols[i].x + COL_WIDTH > (FRAME_WIDTH - BIRD_WIDTH) / 2) {
                nextCol = this.game.cols[i];
                if (i < this.game.cols.length - 1) {
                    nextCol2 = this.game.cols[i+1];
                }
                break;
            }
        }
        let jump = this.controller.getOutputs({
            xDist: (nextCol.x + COL_WIDTH - (FRAME_WIDTH - BIRD_WIDTH) / 2) / FRAME_WIDTH / 2,
            yDist: (nextCol.holeY - this.y) / FRAME_HEIGHT,
            yDist2: (nextCol2.holeY - this.y) / FRAME_HEIGHT,
            velocity: this.velocity
        })['jump'];
        return jump > 0.5;
    }

    /**
     * Update the bird's game state given how much time has passed
     *
     * Parameters:
     *  timeDelta: time in seconds that passed since the last update
     */
    update(timeDelta) {
        if (this.dead) {
            return;
        }

        // The bird jumps if it wants to and enough distance has passed to recharge the jump
        this.x = this.game.x;
        this.y += this.velocity * timeDelta;
        if (this.velocity != TERMINAL_VEL) this.y += GRAVITY * timeDelta * timeDelta / 2;
        if (this.x - this.lastJump > JUMP_THRESHOLD && this.jumps()) {
            this.velocity = -300;
        } else {
            this.velocity = Math.min(this.velocity + GRAVITY * timeDelta, TERMINAL_VEL);
        }


        // Game over if bird hit the top of the screen, bottom of the screen, or a column
        let bbox = {
            x1: (FRAME_WIDTH - BIRD_WIDTH) / 2,
            y1: this.y,
            x2: (FRAME_WIDTH + BIRD_WIDTH) / 2,
            y2: this.y + BIRD_WIDTH
        };
        if (this.y < -BIRD_WIDTH || this.y > FRAME_HEIGHT || this.game.cols.some(col => col.intersects(bbox))) {
            this.dead = true;
            let nextCol;
            for (let col of this.game.cols) {
                if (col.x + COL_WIDTH > (FRAME_WIDTH - BIRD_WIDTH) / 2) {
                    nextCol = col;
                    break;
                }
            }
            this.score = this.game.x - Math.abs(nextCol.holeY - this.y) / FRAME_HEIGHT;
        }
    }

    /**
     * Render this bird onto the context with the given color
     *
     * Parameters:
     *  context: 2d context to render the bird onto
     *  color:   color to render the bird with
     */
    render(context, color) {
        context.fillStyle = color;
        context.fillRect((FRAME_WIDTH - BIRD_WIDTH) / 2 - (this.game.x - this.x), this.y, BIRD_WIDTH, BIRD_WIDTH);
    }
}

/**
 * A column with a hole for birds to go through
 */
class Column {

    /**
     * Creates a column with a hole for birds to go through
     *
     * Parameters:
     *  game:  flappy game this column is a part of
     *  holeY: y position of the center of the hole in this column
     */
    constructor(game, holeY) {
        this.game = game;
        this.startX = this.game.x;
        this.x = this.game.x;
        this.holeY = holeY;
        this.scored = false;
        this.done = false;

        this.topCol = {
            height: holeY - HOLE_HEIGHT / 2,
            width: COL_WIDTH,
            x: FRAME_WIDTH,
            y: 0
        };
        this.botCol = {
            height: FRAME_HEIGHT - holeY - HOLE_HEIGHT / 2,
            width: COL_WIDTH,
            x: FRAME_WIDTH,
            y: this.topCol.height + HOLE_HEIGHT
        }
    }

    /**
     * Finds if this column intersects with the given bounding box
     */
    intersects(bbox) {
        // We have two bounding boxes for this column, the top and bottom, so making a function
        // allows us to find if it intersects with either one easily
        let rectIntersect = function(colBox) {
            if (colBox.x1 >= bbox.x2 || bbox.x1 >= colBox.x2) {
                return false;
            }
            if (colBox.y1 >= bbox.y2 || bbox.y1 >= colBox.y2) {
                return false;
            }
            return true;
        };

        return rectIntersect({
            x1: this.topCol.x,
            y1: this.topCol.y,
            x2: this.topCol.x + this.topCol.width,
            y2: this.topCol.y + this.topCol.height
        }) || rectIntersect({
            x1: this.botCol.x,
            y1: this.botCol.y,
            x2: this.botCol.x + this.botCol.width,
            y2: this.botCol.y + this.botCol.height
        });

    }

    /**
     * Update the position of the column based on the current game state
     */
    update() {
        if (!this.done) {
            let newX = FRAME_WIDTH - this.game.x + this.startX;
            if (!this.scored && newX < FRAME_WIDTH / 2 - COL_WIDTH) {
                this.scored = true;
                this.game.score++;
            }
            if (newX < -COL_WIDTH) {
                this.done = true;
            }
            this.x = newX;
            this.topCol.x = newX;
            this.botCol.x = newX;
        }
    }

    /**
     * Render the column onto the given context
     *
     * Parameters:
     *  context: 2d context to render the column onto
     */
    render(context) {
        context.fillStyle = 'gray';
        context.fillRect(this.topCol.x, this.topCol.y, this.topCol.width, this.topCol.height);
        context.fillRect(this.botCol.x, this.botCol.y, this.botCol.width, this.botCol.height);
    }
}

/**
 * Representation of an entire flappy bird game, with a canvas to render the game onto
 */
class FlappyGame {
    /**
     * Creates the Flappy Game, including the canvas
     *
     * Parameters:
     *  gameContainer: html element to add the canvas to
     */
    constructor(gameContainer) {
        let infobar = document.createElement('div');
        infobar.id = 'info-bar';
        infobar.append(this.scoreDisplay = document.createElement('div'));
        infobar.append(this.highScoreDisplay = document.createElement('div'));
        gameContainer.append(infobar);
        this.scoreDisplay.innerHTML = 'Score: N/A';
        this.highScoreDisplay.innerHTML = 'High Score: N/A';

        this.canvas = document.createElement('canvas');
        this.canvas.id = 'game';
        this.canvas.width = FRAME_WIDTH;
        this.canvas.height = FRAME_HEIGHT;
        gameContainer.append(this.canvas);
        this.context = this.canvas.getContext('2d');

        this.done = true;
        this.highScore = 0;
        this.won = false;
    }

    /**
     * Setup a new round of the game with the given players
     *
     * Parameters:
     *  birds: birds to run the game with
     */
    setup(birds) {
        if (this.done) {
            this.birds = birds;
            birds.forEach(b => b.game = this);
            this.x = 0;
            this.cols = [];
            this.colThreshold = 0;
            this.score = 0;
            this.done = false;
            this.won = false;
        }
    }

    stop() {
        this.done = true;
    }

    /**
     * Update the game state given how much time has passed
     *
     * Parameters:
     *  timeDelta: time that has passed since the last update in seconds
     */
    update(timeDelta) {
        // Break up large time deltas into smaller steps to guarantee that the trajectory of
        // each of the birds is correct
        while (timeDelta > GAME_STEP_S) {
            this.update(GAME_STEP_S);
            timeDelta -= GAME_STEP_S;
        }
        if (this.done) {
            return
        }

        // The bird moves at a constant rate, with columns at fixed distances
        this.x += timeDelta * MOVE_SPEED;
        if (this.x > this.colThreshold) {
            this.colThreshold += COL_DISTANCE;
            this._addColumn();
        }
        this.cols.forEach(col => col.update());
        this.cols = this.cols.filter(col => !col.done);

        this.birds.forEach(b => b.update(timeDelta));

        if (this.birds.every(b => b.dead)) {
            this.done = true;
        } else {
            if (this.score > this.highScore) {
                this.highScore = this.score;
                if (this.score >= MAX_SCORE) {
                    this.done = true;
                    this.won = true;
                }
            }
        }
    }

    /**
     * Render the current game state to the context of this game
     *
     * Parameters:
     *  paused: if true, will render a gray layer over the state to make it obvious it is paused
     */
    render(paused = false) {
        if (this.scoreDisplay.innerHTML != 'Score: ' + this.score) {
            this.scoreDisplay.innerHTML = 'Score: ' + this.score;
        }
        if (this.won) {
            this.highScoreDisplay.innerHTML = 'High Score: MAX';
        } else if (this.highScoreDisplay.innerHTML != 'High Score: ' + this.highScore) {
            this.highScoreDisplay.innerHTML = 'High Score: ' + this.highScore;
        }

        this.context.fillStyle = 'skyblue';
        this.context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);

        this.cols.forEach(col => col.render(this.context));
        this.birds.forEach(b => b.render(this.context, 'yellow'));

        if (paused) {
            this.context.globalAlpha = 0.25;
            this.context.fillStyle = 'grey';
            this.context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
            this.context.globalAlpha = 1;
        }
    }

    /**
     * Add a column to just to the right of the frame, with a randomly placed hole
     */
    _addColumn() {
        this.cols.push(new Column(this,
            Math.floor(Math.random() * (FRAME_HEIGHT * .75 - HOLE_HEIGHT)) + 0.125 * FRAME_HEIGHT + HOLE_HEIGHT / 2));
    }
}

///////////////////////////////////////////
// Network Simulation and Training Class //
///////////////////////////////////////////

/**
 * Simulator class to handle running the multiple generations of AI in the flappy bird game
 */
class FlappySim {

    /**
     * Creates the flappy bird simulator
     */
    constructor() {
        // All of the default options for the simulation
        this.hiddenShape = [8];
        this.inputs = {
            xDist: true,
            yDist: true,
            yDist2: true,
            velocity: false
        };
        this.maxMutationChange = 0.1;
        this.mutationChance    = 0.2;
        this.numBirds          = 100;
        this.numGens           = 10000;
        this.running           = false;
        this.paused            = false;
        this.simRenderInterval = 10;

        let gameContainer = document.getElementById('game-container');
        this.game = new FlappyGame(gameContainer);

        this.genDisplay = document.getElementById('gen-display');
        this.genDisplay.innerHTML = 'Generation: N/A';

        this._addNetControlEventListeners();
        this._addTrainingControlEventListeners();
        this._addSimulationControlEventListeners();
    }

    /**
     * Add the neural network controls to the container, those being the number of hidden layers,
     * how many neurons are in each layer, and what inputs are passed to the network
     *
     * Parameters:
     *  container: element to add network controls to
     */
    _addNetControlEventListeners() {
        let self = this;

        // Control the number of neurons in each layer
        let perLayerSet = document.getElementById('neuron-layer-inputs');
        perLayerSet.setLayers = function(numLayers) {
            if (self.hiddenShape.length > numLayers) {
                self.hiddenShape = self.hiddenShape.slice(0, numLayers);
            } else {
                let newVals = Array(numLayers - self.hiddenShape.length);
                newVals.fill(DEFAULT_NEURONS_PER_LAYER);
                self.hiddenShape.push(...newVals);
            }

            perLayerSet.innerHTML = '';
            for (let i = 1; i <= numLayers; i++) {
                let label = document.createElement('label');
                label.setAttribute('for', `net-layer-${i}`);
                label.innerHTML = `Layer ${i}:`;
                perLayerSet.append(label);

                let input = document.createElement('input');
                input.id = `net-layer-${i}`;
                input.setAttribute('type', 'number');
                input.setAttribute('min', 1);
                input.setAttribute('max', MAX_NEURONS_PER_LAYER);
                input.addEventListener('change', function(e) {
                    clampNumberInput(e.target);
                    self.hiddenShape[i-1] = e.target.valueAsNumber;
                });
                input.value = self.hiddenShape[i-1];
                perLayerSet.append(input);
            }
        };
        perLayerSet.setLayers(this.hiddenShape.length);

        // Control the number of layers
        let netNumLayers = document.getElementById('net-num-layers');
        netNumLayers.value = this.hiddenShape.length;
        netNumLayers.addEventListener('change', function (e) {
            clampNumberInput(e.target);
            perLayerSet.setLayers(e.target.valueAsNumber);
        });


        for (let option of document.querySelectorAll('#net-inputs > input')) {
            if (this.inputs[option.value]) {
                option.checked = true;
            }
            option.addEventListener('change', function(e) {
                self.inputs[e.target.value] = e.target.checked;
            });
        }
    }

    /**
     * Add the neural network training controls to the given container, those being how many models are trained
     * per generation, how likely a mutation will occur per weight/bias in the AI, the maximum change to any
     * weight or bias, and the number of generations to run
     *
     * Parameters:
     *  container: element to add the training controls to
     */
    _addTrainingControlEventListeners() {
        let self = this;

        let simPopSize = document.getElementById('sim-pop-size');
        simPopSize.value = this.numBirds;
        simPopSize.addEventListener('change', function (e) {
            clampNumberInput(e.target);
            self.numBirds = e.target.valueAsNumber;
        });

        let simNumGens = document.getElementById('sim-num-gens')
        simNumGens.value = this.numGens;
        simNumGens.addEventListener('change', function (e) {
            clampNumberInput(e.target);
            self.numGens = e.target.valueAsNumber;
        });

        let maxMutationDisplay = document.getElementById('net-max-mutation-display');
        let netMaxMutation = document.getElementById('net-max-mutation');
        netMaxMutation.value = this.maxMutationChange;
        netMaxMutation.addEventListener('input', function(e) {
            maxMutationDisplay.innerHTML = e.target.valueAsNumber;
            self.maxMutationChange = e.target.valueAsNumber;
        });

        let mutationChanceDisplay = document.getElementById('net-mutation-chance-display');
        let netMutationChance = document.getElementById('net-mutation-chance');
        netMutationChance.value = this.mutationChance;
        netMutationChance.addEventListener('input', function(e) {
            mutationChanceDisplay.innerHTML = e.target.valueAsNumber;
            self.mutationChance = e.target.valueAsNumber;
        });
    }

    /**
     * Add the simulation controls to the container, those being when to start/pause/reset the simulation,
     * and how many generations to skip rendering in full
     *
     * Parameters:
     *  container: element to add the simulation controls to
     */
    _addSimulationControlEventListeners() {
        let self = this;

        let simRenderInterval = document.getElementById('sim-render-interval');
        simRenderInterval.value = this.simRenderInterval;
        simRenderInterval.addEventListener('change', function(e) {
            clampNumberInput(e.target);
            self.simRenderInterval = e.target.valueAsNumber;
        });

        let startButton = document.getElementById('sim-start');
        let pauseButton = document.getElementById('sim-pause');
        pauseButton.style.display = 'none';
        let resumeButton = document.getElementById('sim-resume');
        resumeButton.style.display = 'none';
        let resetButton = document.getElementById('sim-reset');
        resetButton.style.display = 'none';

        startButton.addEventListener('click', function(e) {
            if (!Object.values(self.inputs).some(v => v)) {
                alert('There needs to be at least one input to the bird controller!');
                return;
            }
            document.querySelectorAll('.js_disable-on-start input').forEach(el => el.disabled = true);
            this.style.display = 'none';
            pauseButton.style.display = '';

            self.genDisplay.style.display = '';

            self.start();
        });
        pauseButton.addEventListener('click', function(e) {
            this.style.display = 'none';
            resumeButton.style.display = '';
            resetButton.style.display = '';

            self.paused = true;
        });
        resumeButton.addEventListener('click', function(e) {
            this.style.display = 'none';
            pauseButton.style.display = '';
            resetButton.style.display = 'none';

            self.paused = false;
            setTimeout(self.resumeCallback.bind(self), 1);
        });
        resetButton.addEventListener('click', function(e) {
            document.querySelectorAll('#game-container input').forEach(el => el.disabled = false);
            this.style.display = 'none';
            resumeButton.style.display = 'none';
            startButton.style.display = '';

            self.genDisplay.style.display = 'none';

            self.running = false;
            self.paused = false;
        });
    }

    /**
     * Start the simulation
     */
    start() {
        if (!this.running) {
            this.birds   = [];
            this.currGen = 0;
            this.running = true;
            this.game.stop();
            this._runNextGen();
        }
    }

    /**
     * Run the next generation of the simulation
     */
    _runNextGen() {
        this.currGen++;
        this.genDisplay.innerHTML = 'Generation: ' + this.currGen;
        if (this.game.won || this.currGen > this.numGens) {
            this.running = false;
            this.game.render();
            return;
        }

        let renderGame = this.currGen == 1 ||
                         this.currGen == this.numGens ||
                         this.currGen % this.simRenderInterval == 0;
        this._simulateGame(renderGame);
    }

    /**
     * Simulate a single game of flappy bird
     *
     * Parameters:
     *  render: render the game in full, or skip to every 100000th point
     */
    _simulateGame(render) {
        let self = this;

        // Setup birds
        if (this.birds.length) {
            this.birds = Bird.nextGen(this.birds, this.maxMutationChange, this.mutationChance);
        } else {
            for (let i = 0; i < this.numBirds; i++) {
                this.birds.push(
                    Bird.newRandom(Object.entries(this.inputs).filter(p => p[1]).map(p => p[0]), this.hiddenShape)
                );
            }
        }
        this.game.setup(this.birds);

        if (render) {
            this.speed = 1;
            this.lastTime = null;
            requestAnimationFrame(this._update.bind(this));
        } else {
            let runManyUpdates = function(numUpdates) {
                let prevScore = self.game.score;
                for (let i = 0; i < numUpdates && !self.game.done; i++) {
                    if (self.paused) {
                        self.game.render(true);
                        self.resumeCallback = () => runManyUpdates(numUpdates - i);
                        return;
                    }
                    self.game.update(GAME_STEP_S);
                    if (RENDER_ON_SCORE_REACHED.includes(self.game.score) && prevScore != self.game.score) {
                        self.game.render();
                    }
                    prevScore = self.game.score;
                }

                if (!self.game.done) {
                    setTimeout(() => runManyUpdates(100000), 1);
                } else {
                    self.game.render();
                    setTimeout(self._runNextGen.bind(self), 1);
                }
            };
            setTimeout(() => runManyUpdates(100000), 1);
        }
    }

    /**
     * Update the flappy bird game this simulator is running, based on the time elapsed
     * since the last update. This is used to render the game smoothly
     *
     * Parameters:
     *  timestamp: high resolution current time to get time difference from
     */
    _update(timestamp) {
        if (this.paused) {
            this.lastTime = null;
            this.resumeCallback = () => requestAnimationFrame(this._update.bind(this));
            this.game.render(true);
            return;
        }

        // Calculate the time delta (updates are based on time rather than frames)
        if (!this.lastTime) {
            this.lastTime = timestamp;
            this.speedupTime = timestamp + 5000;
        }
        let timeDelta = (timestamp - this.lastTime) / 1000;
        this.lastTime = timestamp;
        if (timeDelta > 1) {
            requestAnimationFrame(this._update.bind(this));
            return;
        }

        if (timestamp > this.speedupTime) {
            this.speed *= 2;
            this.speedupTime += 5000;
        }

        for (let i = 0; i < this.speed; i++) {
            this.game.update(timeDelta);
        }
        this.game.render();

        if (!this.game.done) {
            requestAnimationFrame(this._update.bind(this));
        } else {
            setTimeout(this._runNextGen.bind(this), 1);
        }
    }
}

//////////////////////////
// Main Execution Point //
//////////////////////////

document.addEventListener("DOMContentLoaded", () => {
    let simulation = new FlappySim();
});
