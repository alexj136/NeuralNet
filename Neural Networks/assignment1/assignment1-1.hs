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
-- (the [LabelledInstance]s), and a learning rate. The output is a function from
-- an instances to classes. We also take an integer parameter that is the number
-- of iterations we've done, and return this in a tuple with the learned
-- function.
{--
learnPerceptron :: Int -> Weights -> [LabelledInstance] -> Float -> (Int, (Instance -> Class))
learnPerceptron iterNo weights trainingInstances learningRate =
    if weights == newWeights then weights
    else learnPerceptron (iterNo + 1) newWeights trainingInstances learningRate
    where
    newWeights :: Weights
    newWeights = someFunction
--}

-- Implementation of the Perceptron Criterion error function
--errorPerc :: Weights -> [ClassifiedInstance] -> Float
errorPerc weights classifiedInstances = (-1) *
    ( sum
        ( ( \x -> fst x * snd x )
            ( zip
                ( map (applyWeights weights)
                    ( map getInstance wronglyClassifiedInstances )
                )
                ( map toInt ( snd wronglyClassifiedInstances )
                )
            )
        )
    )
    where
    wronglyClassifiedInstances :: [ClassifiedInstance]
    wronglyClassifiedInstances = filter wronglyClassified classifiedInstances

-- Apply a weight vector to an instance - essentially the dot product for two
-- vectors.
applyWeights :: Weights -> Instance -> Float
applyWeights w i = (+)
    ((fst w) * (fromIntegral (fst i)))
    ((snd w) * (fromIntegral (snd i)))
