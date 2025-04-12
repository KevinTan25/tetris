package src.pas.tetris.agents;


import java.util.ArrayList;
// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


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


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.99;
    public static final int NUM_ROWS = Board.NUM_ROWS;
    public static final int NUM_COLS = Board.NUM_COLS;

    private Random random;

    private int numberOfHoles;
    private int colDiff;
    private int stackHeight;
    private int maxHeight;
    private int actionClear;
    private double sumOfDifferencesWithAvg;
    private double flatness;
    private boolean reachedTen = false;
    private boolean scoredPoint = false;
    private int wellDepth;

    private int aggregateHeight;
    private int bumpiness;
    private int overhangs;
    
    private int colHeight;
    private int setup;
    private int depth;
    

    public TetrisQAgent(String name)
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


        final int numInputs = 227; // 7 features (flatness, stackHeight, etc.)
        // final int hiddenLayer1 = 256; // First hidden layer size
        // final int hiddenLayer2 = 128; // Second hidden layer size
        
        final int hiddenLayer1 = 128;
        final int hiddenLayer2 = 64;
        // final int hiddenLayer3 = 64;
        final int outputDim = 1; // Q-value as scalar output

        // Create a feedforward neural network
        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numInputs, hiddenLayer1)); // First hidden layer
        qFunction.add(new ReLU()); // Activation function
        qFunction.add(new Dense(hiddenLayer1, hiddenLayer2)); // Second hidden layer
        qFunction.add(new ReLU()); // Activation function
        // qFunction.add(new Dense(hiddenLayer2, hiddenLayer3)); // Third layer
        // qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenLayer2, outputDim)); // Output layer

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
        // Matrix flattenedImage = null;
        // try
        // {
        //     flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
        // } catch(Exception e)
        // {
        //     e.printStackTrace();
        //     System.exit(-1);
        // }
        // return flattenedImage;

        Matrix boardImage = null;
        try {
            boardImage = game.getGrayscaleImage(potentialAction);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        List<Integer> columnHeights = calculateColumnHeights(boardImage); // allCols
        // stackHeight = calculateStack(boardImage); // stack
        maxHeight = columnHeights.stream().max(Integer::compare).orElse(0); 
        int minHeight = columnHeights.stream().min(Integer::compare).orElse(0);
        colDiff = maxHeight - minHeight; // colDiff
        numberOfHoles = calculateHoles(boardImage); // holes
        actionClear = calculateLinesCleared(boardImage); 
        // sumOfDifferencesWithAvg = calculateSumOfDifferences(columnHeights);
        flatness = flatTopReward(boardImage); 


        // aggregateHeight = calculateAggregateHeight(columnHeights);
        // bumpiness = calculateBumpiness(columnHeights);
        // overhangs = calculateOverhangs(boardImage);

        colHeight = calcColsHeights(boardImage);
        setup = evaluateSetup(boardImage);
        depth = depth(boardImage);

        // // Combine the board state with calculated features
        // int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        int numBoardFeatures = Board.NUM_ROWS * Board.NUM_COLS;
        // Matrix input = Matrix.full(1, numPixelsInImage + 7, 0);
        Matrix input = Matrix.full(1, numBoardFeatures + 7, 0);

        // Add custom features
        // int index = numPixelsInImage;
        int index = 0;
        // Normalized inputs
        // input.set(0, index++, stackHeight / (double) NUM_ROWS); // Normalize stack height
        input.set(0, index++, maxHeight);
        input.set(0, index++, numberOfHoles);
        input.set(0, index++, colDiff);
        input.set(0, index++, actionClear);
        // input.set(0, index++, flatness / (double) NUM_COLS);


        // input.set(0, index++, aggregateHeight); // Aggregate height
        // input.set(0, index++, bumpiness); // Bumpiness
        // input.set(0, index++, overhangs); // Overhangs


        input.set(0, index++, colHeight);
        input.set(0, index++, setup);
        input.set(0, index++, depth);

        return input;
    }

    private int calculateAggregateHeight(List<Integer> columnHeights) {
        return columnHeights.stream().mapToInt(Integer::intValue).sum();
    }

    private int calculateBumpiness(List<Integer> columnHeights) {
        int bumpiness = 0;
        for (int i = 0; i < columnHeights.size() - 1; i++) {
            bumpiness += Math.abs(columnHeights.get(i) - columnHeights.get(i + 1));
        }
        return bumpiness;
    }
    
    private int calculateOverhangs(Matrix board) {
        int overhangs = 0;
        for (int col = 0; col < NUM_COLS; col++) {
            boolean blockFound = false;
            for (int row = 0; row < NUM_ROWS; row++) {
                if (board.get(row, col) != 0.0) {
                    blockFound = true; // Found a block
                } else if (blockFound) {
                    // Count all subsequent blocks as overhangs
                    for (int k = row + 1; k < NUM_ROWS; k++) {
                        if (board.get(k, col) != 0.0) {
                            overhangs++;
                        }
                    }
                    break;
                }
            }
        }
        return overhangs;
    }

    private int calculateWellDepth(List<Integer> columnHeights) {
        int wellDepth = 0;
    
        for (int i = 0; i < columnHeights.size(); i++) {
            int leftHeight = (i == 0) ? Integer.MAX_VALUE : columnHeights.get(i - 1);
            int rightHeight = (i == columnHeights.size() - 1) ? Integer.MAX_VALUE : columnHeights.get(i + 1);
            int currentHeight = columnHeights.get(i);
    
            // Check if the current column is a well
            if (currentHeight < leftHeight && currentHeight < rightHeight) {
                wellDepth += Math.min(leftHeight, rightHeight) - currentHeight;
            }
        }
    
        return wellDepth;
    }

    private int calculateHoles(Matrix board) {
        int holes = 0;
        for (int col = 0; col < NUM_COLS; col++) {
            boolean blockFound = false;
            for (int row = 0; row < NUM_ROWS; row++) {
                if (board.get(row, col) != 0.0) {
                    blockFound = true;
                } else if (blockFound) {
                    holes++;
                }
            }
        }
        return holes;
    }

    private List<Integer> calculateColumnHeights(Matrix board) {
        List<Integer> heights = new ArrayList<>();
        for (int col = 0; col < NUM_COLS; col++) {
            int height = 0;
            for (int row = 0; row < NUM_ROWS; row++) {
                if (board.get(row, col) != 0.0) {
                    height = NUM_ROWS - row;
                    break;
                }
            }
            heights.add(height);
        }
        return heights;
    }

    private int calculateLinesCleared(Matrix board) {
        int linesCleared = 0;
        for (int row = 0; row < NUM_ROWS; row++) {
            boolean fullRow = true;
            for (int col = 0; col < NUM_COLS; col++) {
                if (board.get(row, col) == 0.0) {
                    fullRow = false;
                    break;
                }
            }
            if (fullRow) linesCleared++;
        }
        return linesCleared;
    }

    private int evaluateSetup(Matrix gameBoard) {
        int height = 0;
        int previousRowForColumn = -1;
    
        for (int column = NUM_COLS - 1; column >= 0; column--) {
            int emptyCellsInColumn = 0;
            int lastEmptyRow = -1;
    
            for (int row = 0; row < NUM_ROWS; row++) {
                if (gameBoard.get(row, column) == 0.0) {
                    emptyCellsInColumn++;
                    lastEmptyRow = row;
                }
    
                if (emptyCellsInColumn > 1) {
                    height = 0;
                    previousRowForColumn = -1;
                    break;
                } else if (row == NUM_ROWS - 1 && emptyCellsInColumn == 1) {
                    if (lastEmptyRow == previousRowForColumn) {
                        height++;
                    } else {
                        height = 1;
                        previousRowForColumn = row;
                    }
                }
            }
        }
    
        return height;
    }
    
    

    private int calcColsHeights(Matrix gameBoard) {
        int columnHeight = 0;
        int previousRowIndex = -1;
    
        for (int colIndex = NUM_COLS - 1; colIndex > -1; colIndex--) {
            int holeFoundCount = 0;
            int holeRowIndex = -1;
    
            for (int rowIndex = 0; rowIndex < NUM_ROWS; rowIndex++) {
                if (gameBoard.get(rowIndex, colIndex) == 0.0) {
                    holeFoundCount++;
                    holeRowIndex = rowIndex;
                }
    
                if (holeFoundCount > 1) {
                    columnHeight = 0;
                    previousRowIndex = -1;
                    break;
                } else if (rowIndex == NUM_ROWS - 1) {
                    if (holeRowIndex == previousRowIndex) {
                        columnHeight++;
                    } else {
                        columnHeight = 1;
                        previousRowIndex = rowIndex;
                    }
                }
            }
        }
    
        return columnHeight;
    }
    
    private int depth(Matrix gameBoard) {
        int totalDepth = 0;
    
        for (int colIndex = 1; colIndex < NUM_COLS - 1; colIndex++) {
            for (int rowIndex = 0; rowIndex < NUM_ROWS; rowIndex++) {
                if (gameBoard.get(rowIndex, colIndex + 1) != 0 && 
                    gameBoard.get(rowIndex, colIndex) == 0 && 
                    gameBoard.get(rowIndex, colIndex - 1) != 0) {
                    totalDepth += NUM_ROWS - rowIndex;
                }
            }
        }
    
        return totalDepth;
    }
    

    private double calculateSumOfDifferences(List<Integer> columnHeights) {
        double avgHeight = columnHeights.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        return columnHeights.stream()
            .mapToDouble(height -> Math.abs(height - avgHeight))
            .sum();
    }
    
    private double flatTopReward(Matrix board) {
        int playfieldHeight = NUM_ROWS;
        int lowerRegionHeight = playfieldHeight / 2; // Middle point of the playfield
        double rewardMultiplier = 2.0; // Reward multiplier for flatness
    
        double reward = 0.0;
        int currentHeight = calculateStackHeight(board);
        int flatness = calculateFlatness(board);
    
        // If the current height is below the middle point, penalize flat stacks
        if (currentHeight < lowerRegionHeight) {
            double flatnessPenalty = rewardMultiplier / (flatness != 0 ? flatness : 1); // Avoid division by zero
            reward -= flatnessPenalty;
        } else {
            double flatnessReward = (flatness * rewardMultiplier) / 2;
            reward += flatnessReward;
        }
    
        return reward;
    }
    
    private int calculateStackHeight(Matrix board) {
        int stackHeight = 0;
        for (int row = 0; row < NUM_ROWS; row++) {
            for (int col = 0; col < NUM_COLS; col++) {
                if (board.get(row, col) != 0.0) {
                    stackHeight = NUM_ROWS - row;
                    return stackHeight; // Return the height of the first non-empty row
                }
            }
        }
        return stackHeight;
    }
    
    private int calculateFlatness(Matrix board) {
        List<Integer> columnHeights = calculateColumnHeights(board);
        if (columnHeights.isEmpty()) {
            return 0;
        }
        return calculateHeightDifference(columnHeights);
    }

    private int calculateHeightDifference(List<Integer> columnHeights) {
        int maxHeight = columnHeights.stream().max(Integer::compare).orElse(0);
        int minHeight = columnHeights.stream().min(Integer::compare).orElse(0);
        return maxHeight - minHeight;
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
        // return this.getRandom().nextDouble() <= EXPLORATION_PROB;

        double decayFactor = -0.0005;
        double noisyness = 0.05 * (random.nextDouble() - 0.5);
        double explorationProb = EXPLORATION_PROB * Math.exp(decayFactor * gameCounter.getCurrentGameIdx());
        double probs = Math.max(0.2, explorationProb + noisyness);

        return this.random.nextDouble() < probs;
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
        // int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        // return game.getFinalMinoPositions().get(randIdx);

        List<Mino> possibleMoves = game.getFinalMinoPositions();

        if (possibleMoves.isEmpty()) {
            return null; // No moves available
        }

        Mino bestMove = null;
        int maxLinesCleared = 0;

        double randomChoice = random.nextDouble();
        
        // if (randomChoice < 0.5) {
            for (Mino move : possibleMoves) {
                Matrix boardAfterMove = null;
                try {
                    boardAfterMove = game.getGrayscaleImage(move);
                } catch (Exception e) {
                    e.printStackTrace();
                    continue;
                }

                // Calculate the number of lines cleared by this move
                int linesCleared = calculateLinesCleared(boardAfterMove);

                // Update the best move if this one clears more lines
                if (linesCleared > maxLinesCleared) {
                    maxLinesCleared = linesCleared;
                    bestMove = move;
                }
            }
            if (bestMove != null) {
                return bestMove;
            }
        // }
        // if (randomChoice < 0.8) {
            // Evaluate moves using heuristics
            double bestScore = Double.NEGATIVE_INFINITY;

            for (Mino move : possibleMoves) {
                // Get the board state after placing this Mino
                Matrix boardAfterMove = null;
                try {
                    boardAfterMove = game.getGrayscaleImage(move);
                } catch (Exception e) {
                    e.printStackTrace();
                    continue;
                }

                // Calculate heuristic score for the move
                double score = evaluateMove(boardAfterMove);

                // Update the best move if this one is better
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
            }
            
            if (bestMove != null) {
                return bestMove;
            }
        // }

        // If no move is evaluated as the best, fallback to a random move
        int randomIndex = this.random.nextInt(possibleMoves.size());
        // System.out.println("Choosing random move");
        return possibleMoves.get(randomIndex);
    }

    private double evaluateMove(Matrix board) {
        // Calculate heuristic features
        int numHoles = calculateHoles(board);
        int linesCleared = calculateLinesCleared(board);
        int stackHeight = calculateStackHeight(board);
        int flatness = calculateFlatness(board);
        int height = calculateStackHeight(board);
        int newDepth = depth(board);
        int evalSetups = evaluateSetup(board);
        int cols = calcColsHeights(board);

    
        // Heuristic weights (adjust these to tune the strategy)
        double holeWeight = -3.0; // Penalize holes
        double lineClearWeight = 10.0; // Reward clearing lines
        double stackHeightWeight = -2.0; // Penalize high stacks
        double flatnessWeight = 1.5; // Reward flatness
        double heightWeight = 0;
        double depthWeight = 2.0;
        double setupWeight = 2.0;
        double colsWeight = 2.0;
        if (height > 6) {
        heightWeight = -3; }
    
        // Calculate the overall score
        double score = (holeWeight * numHoles) +
                       (lineClearWeight * linesCleared) +
                       (stackHeightWeight * stackHeight) +
                       (flatnessWeight * flatness) + 
                       (height * heightWeight) + 
                       (newDepth * depthWeight) + 
                       (evalSetups * setupWeight) +
                       (colsWeight * cols);
        
        return score;
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
        // return game.getScoreThisTurn();
        Board board = game.getBoard();

        double reward = 0.0;
        reward += 10.0 * game.getScoreThisTurn();
        if (game.getScoreThisTurn() >= 10) {
            reward += 100.0;
            reachedTen = true;
            System.out.println("Scored 10 Points");
        }
        reward += actionClear * 5.0;
        reward += 0.5 * flatness; // Reward for achieving flatness

        reward += 0.8 * setup;
        reward += 0.5 * colHeight;
        reward += 0.4 * depth;
        
        // Negative penalties
        reward -= 0.2 * numberOfHoles; // Penalize holes
        reward -= 0.3 * colDiff; // Penalize column height difference
        reward -= 0.3 * maxHeight; // Penalize tall stacks

        return reward;
    }

}
