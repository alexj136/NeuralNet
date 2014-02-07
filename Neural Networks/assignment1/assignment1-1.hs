-- 2-class problem, so we have a positive and a negative class
data Class = Pos | Neg

-- Our inputs are pairs of integers, an X and a Y value
type Input = (Int, Int)

xval, yval :: Input -> Int
xval = fst
yval = snd
