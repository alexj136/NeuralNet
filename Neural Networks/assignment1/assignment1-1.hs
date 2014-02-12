import Data.List (transpose)

-- 2-class problem, so we have a positive and a negative class
data Class = Pos | Neg
    deriving Eq

instance Show Class where
    show Pos = "+"
    show Neg = "-"

toFloat :: Class -> Float
toFloat Pos = 1.0
toFloat Neg = -1.0

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
    Int                -> -- The number of iterations (0 from external calls)
    Weights            -> -- The initial weights
    [LabelledInstance] -> -- The instances with labels
    Float              -> -- The learning rate
    [Float]            -> -- The error in the previous recursive call, which
                          -- is used to calculate the change in error
        (Int, Weights)    -- Return value is the number of iterations and the
                          -- learned weights
gradDescentB iterNo weights labelledInstances learnRate prevError
    | sum changeInError < 0.001 && sum changeInError > -0.001 =
        (iterNo, weights)
    | otherwise =
        gradDescentB (iterNo + 1) newWeights labelledInstances
            learnRate currentError
    where
    classifiedInstances :: [ClassifiedInstance]
    classifiedInstances =
        map (classifyInstance newWeights) labelledInstances
    newWeights :: Weights
    newWeights = zipWith (\w e -> w - (learnRate * e)) weights changeInError
    currentError, changeInError :: [Float]
    currentError = errorPC weights labelledInstances
    changeInError = zipWith (-) currentError prevError

-- Implementation of the Perceptron Criterion error function
errorPC :: Weights -> [LabelledInstance] -> [Float]
errorPC weights labelledInstances = map ((*) (-1))
    ( map sum
        ( map ( map ( \x -> fst x * (fst (snd x)) * ((fromIntegral . snd . snd) x) ) )
            ( map passToAll
                ( zip weights
                    ( map
                        ( zip ( map toFloat ( map snd wronglyClassifiedInstances ) ) )
                        allXiVals
    )))))
    where
    allXiVals :: [[Int]]
    allXiVals =
        biasVector : transpose (map getInstFromCI wronglyClassifiedInstances)

    wronglyClassifiedInstances :: [ClassifiedInstance]
    wronglyClassifiedInstances = filter wronglyClassified (map (\x -> (x, getClassOfInstance weights (getInstFromLI x))) labelledInstances)

    biasVector :: [Int] -- All values in biasVector are 1
    biasVector = [1 | _ <- [1..(length labelledInstances)]]

    -- Takes a tuple containing a single a, and a list of bs, and returns a list
    -- of tuples where the first element is that a, and the second element is
    -- the nth b element
    passToAll :: (a, [b]) -> [(a, b)]
    passToAll (elem, lst) = map (\y -> (elem, y)) lst

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
