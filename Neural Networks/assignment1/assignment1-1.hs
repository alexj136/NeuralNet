-- 2-class problem, so we have a positive and a negative class
data Class = Pos | Neg
    deriving Eq

instance Show Class where
    show Pos = "+"
    show Neg = "-"

toInt :: Class -> Int
toInt Pos = 1
toInt Neg = -1

-- Our inputs are lists of integers, which for this assignment are always two
-- elements long
type Instance = [Int]

-- retrieve the x1 and x2 values from an instance
x1val, x2val :: Instance -> Int
x1val i = i !! 0
x2val i = i !! 1

-- Our weights are lists of floats, the first element being the bias and the
-- following elements being the weights of the inputs
type Weights = [Float]

-- Retrieve the w1, w2 and bias values from a Weights datum
bias, w1val, w2val :: Weights -> Float
bias w  = w !! 0
w1val w = w !! 1
w2val w = w !! 2

-- Represents a labelled data instance
type LabelledInstance = (Instance, Class)

-- Retrieve the underlying Instance from a LabelledInstance
getInstFromLI :: LabelledInstance -> Instance
getInstFromLI = fst

-- Represents a labelled data instance that has been assigned a class by a
-- classifier
type ClassifiedInstance = (LabelledInstance, Class)

-- Check if a classified instance has been classified incorrectly
wronglyClassified :: ClassifiedInstance -> Bool
wronglyClassified i = snd i /= snd (fst i)

-- Retrieve the underlying instance from within a ClassifiedInstance
getInstFromCI :: ClassifiedInstance -> Instance
getInstFromCI = fst . fst

trainingData :: [Instance]
trainingData = [[x1, x2] | x1 <- [0, 1], x2 <- [0, 1]]

labelCombos :: [[Class]]
labelCombos = let pn = [Pos, Neg] in [[a, b, c, d] |
    a <- pn, b <- pn, c <- pn, d <- pn,
    length ((filter (== Pos)) [a, b, c, d]) == 2]

possibleLabellings :: [[LabelledInstance]]
possibleLabellings = map (zip trainingData) labelCombos

-- Perform gradient descent learning (batch version)
gradDescentB ::
    Int                  -> -- The number of iterations (0 from external calls)
    Weights              -> -- The initial weights
    [ClassifiedInstance] -> -- The instances with labels and initial classes
    Float                -> -- The learning rate
    Float                -> -- The error in the previous recursive call, which
                            -- is used to calculate the change in error
        (Int, Weights)      -- Return value is the number of iterations and the
                            -- learned weights
gradDescentB iterNo weights classifiedInstances learnRate prevError
    | changeInError < 0.00001 && changeInError > -0.00001 =
        (iterNo, weights)
    | otherwise =
        gradDescentB (iterNo + 1) newWeights newClassifiedInstances
            learnRate currentError
    where
    newClassifiedInstances :: [ClassifiedInstance]
    newClassifiedInstances =
        map (classifyInstance newWeights) (map fst classifiedInstances)
    newWeights :: Weights
    newWeights = map (\x -> x - (learnRate * changeInError)) weights
    currentError, changeInError :: Float
    currentError = errorPC weights classifiedInstances
    changeInError = currentError - prevError

-- Implementation of the Perceptron Criterion error function
errorPC :: Weights -> [ClassifiedInstance] -> Float
errorPC weights classifiedInstances = (-1) *
    ( sum
        ( map ( \x -> fst x * fromIntegral (snd x) )
            ( zip
                ( map (applyWeights weights)
                    ( map getInstFromCI wronglyClassifiedInstances )
                )
                ( map toInt ( map snd wronglyClassifiedInstances )
    ))))
    where
    wronglyClassifiedInstances :: [ClassifiedInstance]
    wronglyClassifiedInstances = filter wronglyClassified classifiedInstances


-- Make a ClassifiedInstance from a LabelledInstance by determining its Class
-- from the given weights, and returning the LabelledInstance & Class together
classifyInstance :: Weights -> LabelledInstance -> ClassifiedInstance
classifyInstance w li = (li, (getClassOfInstance w (fst li)))

-- Determine the class of an Instance based on the given weights
getClassOfInstance :: Weights -> Instance -> Class
getClassOfInstance w x
    | (applyWeights w x) >= 0 = Pos
    | otherwise               = Neg

-- Apply a weight vector to an instance - essentially the dot product for two
-- vectors.
applyWeights :: Weights -> Instance -> Float
applyWeights w x
    | length w /= 1 + length x =
        error "Mismatch in dimensions of weights & values"
    | otherwise = (bias w) +
        ( sum
            ( map
                ( \x -> (fst x) * fromIntegral (snd x) )
                ( zip (tail w) x )
        ))
