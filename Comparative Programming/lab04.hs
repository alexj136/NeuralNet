cury :: ((a, b) -> c) -> a -> b -> c
cury f = curriedF where curriedF x y = f (x, y) 

uncury :: (a -> b -> c) -> ((a, b) -> c)
uncury f = uncurriedF where uncurriedF (x, y) = f x y

addOneToAll :: Num a => [a] -> [a]
addOneToAll []    = []
addOneToAll (h:t) = (h + 1) : addOneToAll t

addOneToAll' :: Num a => [a] -> [a]
addOneToAll' xs = [1 + x | x <- xs]

timesTwoAll :: Num a => [a] -> [a]
timesTwoAll []    = []
timesTwoAll (h:t) = (h * 2) : timesTwoAll t

addOne :: Num a => a -> a
addOne = (+) 1

timesTwo :: Num a => a -> a
timesTwo = (*) 2

addOneToAll'' :: Num a => [a] -> [a]
addOneToAll'' = map addOne

timesTwoAll' :: Num a => [a] -> [a]
timesTwoAll' = map timesTwo

map' :: (a -> b) -> [a] -> [b]
map' _ []    = []
map' f (h:t) = (f h) : (map' f t)

boolToBits :: [Bool] -> [Integer]
boolToBits = map (\x -> if x then 1 else 0)

heads :: [[a]] -> [a]
heads = map (\x -> head x)
-- heads = map head

data BinaryTree a = EmptyNode
                  | Node a (BinaryTree a) (BinaryTree a)
    deriving Show

mapTree :: (a -> b) -> BinaryTree a -> BinaryTree b
mapTree _ EmptyNode      = EmptyNode
mapTree f (Node x b1 b2) = Node (f x) (mapTree f b1) (mapTree f b2)
