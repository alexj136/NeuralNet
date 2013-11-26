data MyBool = T | F deriving (Show, Eq)

myAnd :: MyBool -> MyBool -> MyBool
myAnd T T = T
myAnd _ _ = F

myOr :: MyBool -> MyBool -> MyBool
myOr F F = F
myOr _ _ = T

myNot :: MyBool -> MyBool
myNot T = F
myNot F = T

data RPS = Rock | Paper | Scissors deriving (Show, Eq)

beats :: RPS -> RPS -> MyBool
beats Rock     Scissors = T
beats Paper    Rock     = T
beats Scissors Paper    = T
beats _        _        = F

data Nat = Zero | Succ Nat

instance Eq Nat where
    (==) Zero     Zero     = True
    (==) (Succ x) (Succ y) = (==) x y
    (==) _        _        = False

instance Show Nat where
    show x = show (natToInt 0 x)
        where natToInt :: Int -> Nat -> Int
              natToInt acc Zero     = acc
              natToInt acc (Succ x) = natToInt (acc + 1) x

add :: Nat -> Nat -> Nat
add x Zero     = x
add x (Succ y) = add (Succ x) y

mult :: Nat -> Nat -> Nat
mult Zero     _ = Zero
mult (Succ x) y = add (mult x y) y
