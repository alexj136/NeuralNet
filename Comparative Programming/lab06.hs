sumAll :: [Int] -> Int
sumAll []    = 0
sumAll (h:t) = h + sumAll t

multAll :: [Int] -> Int
multAll []    = 1
multAll (h:t) = h * multAll t

fold :: (b -> a -> b) -> b -> [a] -> b
fold f x []    = x
fold f x (h:t) = fold f (f x h) t

-- type for fold when used for len:
-- fold :: (Int -> a -> Int) -> Int -> [a] -> Int

len :: [a] -> Int
len = fold (\someInt -> \anything -> someInt + 1) 0

maxi :: Ord a => [a] -> a
maxi []    = error "Empty list has no maximum element"
maxi (h:t) = fold (\a -> \b -> if a < b then b else a) h t

flatten :: [[a]] -> [a]
flatten = fold (++) []

data IntOrBool = AnInt Int | ABool Bool
type ListIntAndBool = [IntOrBool]

data LamExp = Var String | Abs String LamExp | App LamExp LamExp
