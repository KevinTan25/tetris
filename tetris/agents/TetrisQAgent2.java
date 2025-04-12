package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent2
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.99;

    private Random random;

    public static final int NUM_ROWS = Board.NUM_ROWS;
    public static final int NUM_COLS = Board.NUM_COLS;

    private int clears;
    private int maxColHeight;
    private int holes;
    private List<Integer> colHeights;
    private double unevenness;
    private double flatness;
    private int stack;
    private int colDiff;
    private int iColHeight;
    private int setup;
    private int depth;

    public TetrisQAgent2(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int inputDim = 227;
        final int hidden1 = 128;
        final int hidden2 = 64;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputDim, hidden1));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hidden1, hidden2));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hidden2, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix flattenedImage = null;
        try {
            flattenedImage = game.getGrayscaleImage(potentialAction);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        colHeights = allCols(flattenedImage);
        stack = stack(flattenedImage);
        maxColHeight = maxCol(flattenedImage);
        colDiff = heightDiff(colHeights);
        holes = holes(flattenedImage);
        clears = cleared(flattenedImage);
        flatness = flatScore(flattenedImage);
        unevenness = unevenness(flattenedImage);
        iColHeight = iSetup(flattenedImage);
        setup = possibleSetup(flattenedImage);
        depth = depth(flattenedImage);

        int numBoardFeatures = NUM_ROWS * NUM_COLS;
        Matrix inputMat = Matrix.full(1, numBoardFeatures + 7, 0);

        int i = 0;
        inputMat.set(0, i++, maxColHeight);
        inputMat.set(0, i++, holes);
        inputMat.set(0, i++, clears);
        // inputMat.set(0, i++, flatness);
        inputMat.set(0, i++, colDiff);
        // inputMat.set(0, i++, stack);
        // inputMat.set(0, i++, unevenness);
        inputMat.set(0, i++, iColHeight);
        inputMat.set(0, i++, setup);
        inputMat.set(0, i++, depth);

        return inputMat;
    }

    // features
    private int cleared(Matrix image) {
        int numCleared = 0;

        for (int i = 0; i < NUM_ROWS; i++) {
            boolean clearable = true;

            for (int j = 0; j < NUM_COLS; j++) {

                if (image.get(i, j) == 0.0) {
                    clearable = false;
                    break;
                }
            }

            if (clearable) {
                numCleared++;
            }
        }
        
        return numCleared;
    }

    private int maxCol(Matrix image) {
        int maxColHeight = 0;

        for (int i = 0; i < NUM_COLS; i++) {
            int currHeight = 0;

            for (int j = 0; j < NUM_ROWS; j++) {
                if (image.get(j, i) != 0.0) {
                    currHeight = NUM_ROWS - j;

                    if (currHeight > maxColHeight) {
                        maxColHeight = currHeight;
                    }
                }
            }
        }

        return maxColHeight;
    }

    private int holes(Matrix image) {
        int numHoles = 0;

        for (int i = 0; i < NUM_COLS; i++) {
            boolean found = false;

            for (int j = 0; j < NUM_ROWS; j++) {
                if (image.get(j, i) != 0.0) {
                    found = true;
                } else if (found) {
                    numHoles++;
                }
            }
        }

        return numHoles;
    }

    private List<Integer> allCols(Matrix image) {
        List<Integer> colHeights = new ArrayList<>();

        for (int i = 0; i < NUM_COLS; i++) {
            int currHeight = 0;

            for (int j = 0; j < NUM_ROWS; j++) {
                if (image.get(j, i) != 0.0) {
                    currHeight = NUM_ROWS - j;
                    break;
                }
            }

            colHeights.add(currHeight);
        }

        return colHeights;
    }

    private int possibleSetup(Matrix image) {
        int setup = 0;

        for (int i = 0; i < NUM_ROWS; i++) {
            int numHoles = 0;

            for (int j = 0; j < NUM_COLS; j++) {
                if (image.get(i, j) == 0.0) {
                    numHoles++;
                }
            }

            if (numHoles == 2) {
                setup = 1;
            } else if (numHoles == 1) {
                setup = 2;
            }
        }

        return setup;
    }

    private double unevenness(Matrix image) {
        double unevenness = 0.0;
        unevenness = allCols(image).stream().mapToDouble(x -> Math.abs(x - 4)).sum();

        return unevenness;
    }

    private int heightDiff(List<Integer> c) {
        int minHeight = c.stream().min(Integer::compare).orElse(0);
        int maxHeight = c.stream().max(Integer::compare).orElse(0);

        return minHeight - maxHeight;
    }

    private double flatScore(Matrix image) {
        int height = NUM_ROWS / 2; 
        double reward = 0.0;
        int currHeight = stack(image);
        int flatness = flat(image);

        if (currHeight < height) {
            if (flatness != 0) {
                reward -= 2.0 / flatness;
            }
        } else {
            reward += (flatness * 2.0) / 2;
        }

        return reward;
    }

    private int stack(Matrix image) {
        int height = 0;
        
        for (int i = 0; i < NUM_ROWS; i++) {
            for (int j = 0; j < NUM_COLS; j++) {
                if (image.get(i, j) != 0.0) {
                    height = NUM_ROWS - i;

                    return height;
                }
            }
        }
        return height;
    }

    private int iSetup(Matrix image) {
        int singleColHeight = 0;
        int singleIndex = -1;

        for (int i = NUM_COLS - 1; i > -1; i--) {
            int found = 0;
            int foundRowIndex = -1;

            for (int j = 0; j < NUM_ROWS; j++) {
                if (image.get(j, i) == 0.0) {
                    found++;
                    foundRowIndex = j;
                }
                if (found > 1) {
                    singleColHeight = 0;
                    singleIndex = -1;
                    break;
                } else if (j == NUM_ROWS - 1) {
                    if (foundRowIndex == singleIndex) {
                        singleColHeight++;
                    } else {
                        singleColHeight = 1;
                        singleIndex = j;
                    }
                }
            }
        }

        return singleColHeight;
    }

    private int flat(Matrix image) {
        List<Integer> cols = allCols(image);

        if (cols.isEmpty()) {
            return 0;
        }

        return heightDiff(cols);
    }

    private int depth(Matrix image) {
        int depthAmt = 0;

        for (int i = 1; i < NUM_COLS -1; i++) {
            for (int j = 0; j < NUM_ROWS; j++) {
                if (image.get(j, i + 1) != 0 && image.get(j, i) == 0 && image.get(j, i - 1) != 0) {
                    depthAmt += NUM_ROWS - j;
                }
            }
        }

        return depthAmt;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        double noise = 0.05 * (random.nextDouble() - 0.5);
        double p = Math.exp(-0.0005 * gameCounter.getCurrentGameIdx()) * EXPLORATION_PROB;
        double probs = Math.max(0.2, p + noise);

        return probs > random.nextDouble();
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {   
        List<Mino> minos = game.getFinalMinoPositions();
        int clears = -1;
        Mino best = null;

        if (minos.isEmpty()) {
            return null;
        }

        for (Mino mino : minos) {
            Matrix newImage = null;

            try {
                newImage = game.getGrayscaleImage(mino);
            } catch (Exception E) {
                continue;
            }

            int newClears = cleared(newImage);

            if (newClears > clears) {
                best = mino;
                clears = newClears;
            }
        }

        if (best == null) {
            double score = Double.NEGATIVE_INFINITY;

            for (Mino mino : minos) {
                Matrix newImage = null;

                try {
                    newImage = game.getGrayscaleImage(mino);
                } catch (Exception e) {
                    continue;
                }

                int newClears = cleared(newImage);
                int newHoles = holes(newImage);
                int newMaxHeight = maxCol(newImage);
                int newFlatness = flat(newImage);
                int newStack =  stack(newImage);
                int newSetup = possibleSetup(newImage);
                int newISetup = iSetup(newImage);
                int newDepth = depth(newImage);

                double wHeight = 0.0;

                if (newMaxHeight > 6) {
                    wHeight = -3.0;
                }

                double newScore = (10.0 * newClears) + (1.5 * newFlatness) + (2.0 * newDepth) + (-2.0 * newStack) + (-3.0 * newHoles) + (wHeight * newMaxHeight) + (2.0 * newSetup) + (2.0 * newISetup );

                if (newScore > score) {
                    best = mino;
                    score = newScore;
                }
            }

            if (best == null) {
                int randIdx = this.random.nextInt(minos.size());
                best = minos.get(randIdx);
            }
        }

        return best;
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {   
        // if (game.didAgentLose()) {
        //     return -20.0;
        // }

        double reward = 0.0;
        int score = game.getScoreThisTurn();
        
        reward += 10.0 * score;

        // if (score >= 10) {
        //     reward += 50.0;
        // }

        // reward += 20.0 * clears;
        // // reward += 2.0 * flatness;
        // reward += 5.0 * setup; 
        // reward += 2.0 * iColHeight;
        // reward += 3.0 * depth;
        // reward -= 0.3 * holes;
        // reward -= 0.3 * maxColHeight;
        // // reward -= Math.exp(maxColHeight / NUM_ROWS);

        // if (unevenness != 0.0) {
        //     reward -= 0.2 * unevenness;
        // }

        reward += 0.6 * clears;
        reward += 0.5 * flatness;
        reward += 0.8 * setup; 
        reward += 0.5 * iColHeight;
        reward += 0.4 * depth;
        reward -= 0.2 * holes;
        reward -= 0.3 * maxColHeight;
        reward -= 0.3 * colDiff;


        return reward;
    }

}
