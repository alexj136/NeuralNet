-- 2-class problem, so we have a positive and a negative class
data Class = Pos | Neg
    deriving Eq

instance Show Class where
    show Pos = "+"
    show Neg = "-"

-- Our inputs are pairs of integers, an X and a Y value
type Instance = (Int, Int)

-- Our weights are also integers
type Weight = Int

type LabelledInstance = (Instance, Class)

xval, yval :: Instance -> Int
xval = fst
yval = snd

trainingData :: [Instance]
trainingData = [(0, 0), (0, 1), (1, 0), (1, 1)]

possibleLabellings :: [ [ LabelledInstance ] ]
possibleLabellings = map (zip trainingData)
    [ [ Pos, Pos, Neg, Neg ]
    , [ Pos, Neg, Pos, Neg ]
    , [ Pos, Neg, Neg, Pos ]
    , [ Neg, Pos, Pos, Neg ]
    , [ Neg, Pos, Neg, Pos ]
    , [ Neg, Neg, Pos, Pos ]
    ]

-- To learn a perceptron , we need initial weights, some training instances
-- (the [LabelledInstance]s), and a learning rate. The output is a function from
-- an instances to classes. We also take an integer parameter that is the number
-- of iterations we've done, and return this in a tuple with the learned
-- function.
--learnPerceptron :: Int -> [Weight] -> [LabelledInstance] -> Float -> (Int, (Input -> Class))
--learnPerceptron iterNo weights trainingInstances learningRate =
    --let converged = false in
