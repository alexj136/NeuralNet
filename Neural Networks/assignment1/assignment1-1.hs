-- 2-class problem, so we have a positive and a negative class
data Class = Pos | Neg
    deriving Eq

instance Show Class where
    show Pos = "+"
    show Neg = "-"

toInt :: Class -> Int
toInt Pos = 1
toInt Neg = -1

-- Our inputs are pairs of integers, an x1 and an x2 value
type Instance = (Int, Int)

-- retrieve the x1 and x2 values from an instance
x1val, x2val :: Instance -> Int
x1val = fst
x2val = snd

-- Our weights are floats
type Weights = (Float, Float)

-- Represents a labelled data instance
type LabelledInstance = (Instance, Class)

-- Represents a labelled data instance that has been assigned a class by a
-- classifier
type ClassifiedInstance = (LabelledInstance, Class)

-- Check if a classified instance has been classified incorrectly
wronglyClassified :: ClassifiedInstance -> Bool
wronglyClassified i = snd i /= snd (fst i)

-- Retrieve the underlying instance from within a ClassifiedInstance
getInstance :: ClassifiedInstance -> Instance
getInstance = fst . fst

trainingData :: [Instance]
trainingData = [(x1, x2) | x1 <- [0, 1], x2 <- [0, 1]]

labelCombos :: [[Class]]
labelCombos = let pn = [Pos, Neg] in [[a, b, c, d] |
    a <- pn, b <- pn, c <- pn, d <- pn,
    length ((filter (== Pos)) [a, b, c, d]) == 2]

possibleLabellings :: [[LabelledInstance]]
possibleLabellings = map (zip trainingData) labelCombos

-- To learn a perceptron , we need initial weights, some training instances
-- (the [ClassifiedInstance]s), a learning rate, and the error for the previous
-- weights. The output is the learned weights, and the number of iterations. We
-- also take an integer parameter that is the number of iterations we've done,
-- and return this in a tuple with the learned weights.
-- BATCH VERSION
learnPerceptron :: Int -> Weights -> [ClassifiedInstance] ->
    Float -> Float -> (Int, Weights)
learnPerceptron iterNo (w1, w2) classifiedInstances learnRate prevError =
    if changeInError < 0.001 && changeInError > -0.001 then
        (iterNo, (w1, w2))
    else
        learnPerceptron (iterNo + 1) newWeights newClassifiedInstances
            learnRate currentError
    where
    newClassifiedInstances :: [ClassifiedInstance]
    newClassifiedInstances =
        map (classifyInstance newWeights) (map fst classifiedInstances)
    newWeights :: Weights
    newWeights = (
        w1 - (learnRate * changeInError) ,
        w2 - (learnRate * changeInError) )
    currentError, changeInError :: Float
    currentError = errorPerc (w1, w2) classifiedInstances
    changeInError = currentError - prevError

-- Implementation of the Perceptron Criterion error function
errorPerc :: Weights -> [ClassifiedInstance] -> Float
errorPerc weights classifiedInstances = (-1) *
    ( sum
        ( map ( \x -> fst x * fromIntegral (snd x) )
            ( zip
                ( map (applyWeights weights)
                    ( map getInstance wronglyClassifiedInstances )
                )
                ( map toInt ( map snd wronglyClassifiedInstances )
                )
            )
        )
    )
    where
    wronglyClassifiedInstances :: [ClassifiedInstance]
    wronglyClassifiedInstances = filter wronglyClassified classifiedInstances


-- Make a ClassifiedInstance from a LabelledInstance by determining its Class
-- from the given weights, and returning the LabelledInstance & Class together
classifyInstance :: Weights -> LabelledInstance -> ClassifiedInstance
classifyInstance w i = (i, (getClassOfInstance w (fst i)))

-- Determine the class of an Instance based on the given weights
getClassOfInstance :: Weights -> Instance -> Class
getClassOfInstance w i = if (applyWeights w i) >= 0 then Pos else Neg

-- Apply a weight vector to an instance - essentially the dot product for two
-- vectors.
applyWeights :: Weights -> Instance -> Float
applyWeights w i = (+)
    ((fst w) * (fromIntegral (fst i)))
    ((snd w) * (fromIntegral (snd i)))
