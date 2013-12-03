type Name = [Char]
data Proposition = Clause Name [Name]
data Program = Prog [Proposition]

instance Show Proposition where
    show (Clause n l) | length l > 0 = n ++ " :- " ++ showPropList "" l
                      | otherwise    = n ++ "."
        where showPropList str [h]   = str ++ h ++ "."
              showPropList str (h:t) = showPropList (str ++ h ++ ", ") t

instance Show Program where
    show (Prog [])    = ""
    show (Prog (h:t)) = show h ++ "\n" ++ show t

lookup :: Program -> Name -> Maybe [Name]
lookup p n = case p of
    Prog []    -> Nothing
    Prog (h:t) -> case h of
        Clause n' ns | n == n' = 
