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


public class TetrisQAgent3
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.99;
    public static final int NUM_ROWS = Board.NUM_ROWS;
    public static final int NUM_COLS = Board.NUM_COLS;

    private boolean reachedTen = false;
    private int heightDifference;
    List<Integer> columnHeights;
    private int maxHeight;
    private int totalHoles;
    private int linesCleared;
    private double flatness;




    private Random random;

    public TetrisQAgent3(String name)
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
        
        // final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        // final int hiddenDim = 2 * numPixelsInImage;
        // final int outDim = 1;

        // Sequential qFunction = new Sequential();
        // qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        // qFunction.add(new Tanh());
        // qFunction.add(new Dense(hiddenDim, outDim));

        // return qFunction;

        final int numInputs = 5;
        final int hiddenLayer1 = 64; // First hidden layer size
        final int hiddenLayer2 = 32; // Second hidden layer size
        final int outputDim = 1; // Q-value as scalar output

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numInputs, hiddenLayer1)); // First hidden layer
        qFunction.add(new ReLU()); // Activation function
        qFunction.add(new Dense(hiddenLayer1, hiddenLayer2)); // Second hidden layer
        qFunction.add(new ReLU()); // Activation function
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
        // Access the board state after placing the current piece
        Board board = game.getBoard();

        // Calculate features
        List<Integer> columnHeights = calculateColumnHeights(board);
        maxHeight = columnHeights.stream().max(Integer::compare).orElse(0);
        int minHeight = columnHeights.stream().min(Integer::compare).orElse(0);
        heightDifference = maxHeight - minHeight;
        totalHoles = calculateHoles(board);
        linesCleared = calculateLinesCleared(board);
        flatness = calculateFlatness(columnHeights);

        // Normalize and package the features into a matrix
        Matrix input = Matrix.full(1, 5, 0);
        int index = 0;

        // System.out.println("Max height: " + maxHeight + ", Normalized: " + (maxHeight / 20));
        // System.out.println("Hole total: " + totalHoles + ", Normalized: " + (totalHoles / 10));
        // System.out.println("heightDifference: " + heightDifference + ", Normalized: " + (heightDifference / 20));
        // System.out.println("linesCleared: " + linesCleared + ", Normalized: " + (linesCleared / 7));
        // System.out.println("flatness: " + flatness + ", Normalized: " + (flatness / 10));


        input.set(0, index, maxHeight / 10);           // Max height
        input.set(0, index++, totalHoles / 10);         // Total holes
        input.set(0, index++, heightDifference / 10); // Height difference (flatness proxy)
        input.set(0, index++, linesCleared / 7);       // Lines cleared
        input.set(0, index++, flatness / 10);           // Flatness score

        return input;
    }

    /**
     * Calculates the height of each column in the board.
     * @param board The current board state.
     * @return A list of heights for each column.
     */
    private List<Integer> calculateColumnHeights(Board board) {
        List<Integer> columnHeights = new ArrayList<>();
        for (int col = 0; col < NUM_COLS; col++) {
            int height = 0;
            for (int row = 0; row < NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    height = NUM_ROWS - row;
                    break;
                }
            }
            columnHeights.add(height);
        }
        return columnHeights;
    }

    /**
     * Counts the number of holes in the board.
     * @param board The current board state.
     * @return The total number of holes.
     */
    private int calculateHoles(Board board) {
        int holes = 0;
        for (int col = 0; col < NUM_COLS; col++) {
            boolean blockFound = false;
            for (int row = 0; row < NUM_ROWS - 1; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    blockFound = true;
                } else if (blockFound) {
                    holes++;
                }
            }
        }
        return holes;
    }

    /**
     * Calculates the number of full lines that can be cleared.
     * @param board The current board state.
     * @return The number of full lines.
     */
    private int calculateLinesCleared(Board board) {
        int linesCleared = 0;
        for (int row = 0; row < NUM_ROWS; row++) {
            boolean fullRow = true;
            for (int col = 0; col < NUM_COLS; col++) {
                if (!board.isCoordinateOccupied(col, row)) {
                    fullRow = false;
                    break;
                }
            }
            if (fullRow) linesCleared++;
        }
        return linesCleared;
    }

    /**
     * Calculates the flatness of the board by finding the sum of height differences between adjacent columns.
     * @param columnHeights The heights of each column.
     * @return The flatness score.
     */
    private double calculateFlatness(List<Integer> columnHeights) {
        double flatness = 0.0;
        for (int i = 0; i < columnHeights.size() - 1; i++) {
            flatness += Math.abs(columnHeights.get(i) - columnHeights.get(i + 1));
        }
        return flatness;
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

        double decayFactor = 0.0005;
        double explorationProb = EXPLORATION_PROB - (decayFactor * gameCounter.getCurrentGameIdx());

        if (reachedTen == true) {
            return false;
        }

        if (explorationProb < 0.2) {
            return this.getRandom().nextDouble() <= 0.2;
        }

        // Decide whether to explore
        return this.random.nextDouble() < explorationProb;
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
    // @Override
    // public Mino getExplorationMove(final GameView game)
    // {
    //     int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
    //     return game.getFinalMinoPositions().get(randIdx);
    // }
    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> possibleMoves = game.getFinalMinoPositions();

        if (possibleMoves == null || possibleMoves.isEmpty()) {
            return null; // No moves available
        }

        Mino bestMove = null;
        // double bestScore = Double.NEGATIVE_INFINITY;

        // double randomChoice = random.nextDouble();

        // // if (randomChoice < 0.8) {
        //     // Choose a move that clears the most lines
        //     for (Mino move : possibleMoves) {
        //         Board boardAfterMove = simulateBoardAfterMove(game.getBoard(), move);
        //         if (boardAfterMove == null) {
        //             continue; // Skip invalid moves
        //         }

        //         int linesCleared = calculateLinesCleared(boardAfterMove);
        //         if (linesCleared > bestScore) {
        //             bestScore = linesCleared;
        //             bestMove = move;
        //         }
        //     }
        //     if (bestMove != null) {
        //         return bestMove;
        //     }
        // // } else {
        //     // Choose a move using a heuristic evaluation
        //     for (Mino move : possibleMoves) {
        //         Board boardAfterMove = simulateBoardAfterMove(game.getBoard(), move);
        //         if (boardAfterMove == null) {
        //             continue; // Skip invalid moves
        //         }

        //         double score = evaluateMove(boardAfterMove);
        //         if (score > bestScore) {
        //             bestScore = score;
        //             bestMove = move;
        //         }
        //     }
        // // }

        // // If no best move is found, return a random move
        // if (bestMove == null) {
        //     int randomIndex = this.random.nextInt(possibleMoves.size());
        //     return possibleMoves.get(randomIndex);
        // }

        // return bestMove;


        double temperature = 1.0;  // Higher values increase randomness
        double expQValueSum = 0.0;
        for (Mino move : possibleMoves) {
            double qValue = evaluateMove(simulateBoardAfterMove(game.getBoard(), move));
            expQValueSum += Math.exp(qValue / temperature);
        }
        for (Mino move : possibleMoves) {
            double qValue = evaluateMove(simulateBoardAfterMove(game.getBoard(), move));
            double probability = Math.exp(qValue / temperature) / expQValueSum;
            if (this.random.nextDouble() < probability) {
                bestMove = move;
            }
        }

        if (bestMove == null) {
            int randomIndex = this.random.nextInt(possibleMoves.size());
            return possibleMoves.get(randomIndex);
        }

        return bestMove;

    }

    /**
     * Simulates the board state after a specific move.
     * @param board The current board state.
     * @param move The move to simulate.
     * @return The board state after the move, or null if the move is invalid.
     */
    private Board simulateBoardAfterMove(Board board, Mino move) {
        if (board == null || move == null) {
            return null;
        }

        Board boardClone = board;  // Ensure no changes to the original board
        // if (!boardClone.addMino(move)) {  // Simulate placing the Mino
        //     return null; // If the move is invalid, return null
        // }

        boardClone.clearFullLines(); // Simulate clearing lines
        return boardClone;
    }

    /**
     * Evaluates the heuristic score of a board state.
     * @param board The current board state.
     * @return The heuristic score.
     */
    private double evaluateMove(Board board) {
        if (board == null) {
            return Double.NEGATIVE_INFINITY; // Return a very low score for invalid boards
        }

        int numHoles = calculateHoles(board);
        int linesCleared = calculateLinesCleared(board);
        double flatness = calculateFlatness(calculateColumnHeights(board));

        // Heuristic weights
        double holeWeight = -5.0;           // Penalize holes
        double lineClearWeight = 10.0;      // Reward clearing lines
        double flatnessWeight = 1.5;        // Reward flatness

        // Calculate the overall score
        return (holeWeight * numHoles) +
            (lineClearWeight * linesCleared) +
            (flatnessWeight * flatness);
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
        double reward = 0.0;

        reward += 10.0 * game.getScoreThisTurn();
        if (game.getScoreThisTurn() >= 10) {
            reward += 100.0;
            reachedTen = true;
            System.out.println("Scored 10 Points");
        }
        // if (game.getScoreThisTurn() >= 5) {
        //     scoredPoint = true;
        //     // System.out.println("Scored 5 Points");
        // }
        // reward += 0.6 * actionClear; // Reward for clearing lines
        reward += linesCleared * 5.0;
        reward += 2.0 * flatness; // Reward for achieving flatness
        
        // Negative penalties
        reward -= 0.5 * totalHoles; // Penalize holes
        reward -= 2.0 * heightDifference; // Penalize column height difference
        // reward -= 0.3 * sumOfDifferencesWithAvg; // Penalize uneven distribution of heights
        reward -= 3.0 * maxHeight; // Penalize tall stacks


        return reward;
        // return game.getScoreThisTurn();
    }

}
