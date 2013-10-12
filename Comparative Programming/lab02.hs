-- Question 1
inorder :: Ord a => [a] -> Bool
inorder []     = True
inorder [x]    = True
inorder (x:xs) = x <= head xs && inorder xs

insert :: Ord a => a -> [a] -> [a]
insert x [] = [x]
insert x xs | x <= head xs = x:xs
            | otherwise    = [head xs] ++ insert x (tail xs)

sort :: Ord a => [a] -> [a]
sort []    = []
sort (h:t) | inorder (h:t) = (h:t)
           | otherwise     = insert h (sort t)

-- Question 2
data BinaryTree a = EmptyNode
                  | Node a (BinaryTree a) (BinaryTree a)
    deriving Show

toList :: BinaryTree a -> [a]
toList EmptyNode    = []
toList (Node e x y) = (toList x) ++ [e] ++ (toList y)

inorderTree :: Ord a => BinaryTree a -> Bool
inorderTree = inorder . toList

insertTree :: Ord a => a -> BinaryTree a -> BinaryTree a
insertTree e EmptyNode     = Node e EmptyNode EmptyNode
insertTree e (Node e' x y) | e <= e'   = Node e' (insertTree e x) EmptyNode
                           | otherwise = Node e' EmptyNode (insertTree e y)

-- Question 3
preOrderTree :: BinaryTree a -> [a]
preOrderTree EmptyNode = []
preOrderTree (Node e x y) = [e] ++ (toList x) ++ (toList y)

postOrderTree :: BinaryTree a -> [a]
postOrderTree EmptyNode = []
postOrderTree (Node e x y) = (toList x) ++ (toList y) ++ [e]

revList :: [a] -> [a]
revList []     = []
revList (x:xs) = (revList xs) ++ [x]

fromList :: [a] -> BinaryTree a
fromList []  = EmptyNode
fromList [x] = Node x EmptyNode EmptyNode
fromList xs  = Node (head secondX) (fromList firstX) (fromList (tail secondX))
    where (firstX, secondX) = splitAt (length xs `div` 2) xs

revTree :: BinaryTree a -> BinaryTree a
revTree = fromList . revList . toList
