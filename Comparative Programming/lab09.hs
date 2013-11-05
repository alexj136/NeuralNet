data Exp = Var Name | App Exp Exp | Abs Name Exp deriving (Show, Eq)

type Name = String

data Set a = Elem a (Set a) | Empty

instance Show a => Show (Set a) where
    show s = "<" ++ (show2 s) ++ ">"

show2 :: Show a => Set a -> String
show2 Empty          = ""
show2 (Elem x Empty) = show x
show2 (Elem x rest)  = show x ++ ", " ++ show2 rest

add :: Eq a => Set a -> a -> Set a
add Empty          x = Elem x Empty
add (Elem x' rest) x | x == x'   = Elem x' rest
                     | otherwise = add rest x

get :: Set a -> a
get Empty      = error "Tried to get element of empty set!"
get (Elem x _) = x

remove :: Eq a => Set a -> a -> Set a
remove Empty _       = Empty
remove (Elem x rest) x' | x == x' = rest
                        | otherwise = Elem x (remove rest x')

union :: Eq a => Set a -> Set a -> Set a
union s     Empty = s
union Empty s     = s
union s1    s2    = union (add s1 e) (remove s2 e) where e = get s2

isEmpty :: Set a -> Bool
isEmpty Empty = True
isEmpty _     = False

freeVars :: Exp -> Set Name
freeVars x = case x of
    Var v   -> Elem v Empty
    Abs v m -> remove (freeVars m) v
    App m n -> union (freeVars m) (freeVars n)

closed :: Exp -> Bool
closed x = isEmpty $ freeVars x

subst :: Name -> Exp -> Exp -> Exp
subst v body arg = case body of
    Var n   | v == n -> arg
            | v /= n -> Var n
    Abs n x | v == n -> Abs n x
            | v /= n -> Abs n (subst v x arg)
    App m n          -> App (subst v m arg) (subst v n arg)

eval :: Exp -> Exp
eval (App (Abs n x) y) = eval (subst n x y)
eval other             = other
