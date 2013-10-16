import System.Random (randomRIO) -- to choose a pivot

first :: Integer -> [a] -> [a]
first 0 _     = []
first 1 (h:t) = [h]
first n []    = error "You got greedy"
first n (h:t) = h : first (n - 1) t

filt :: Integer -> [Integer] -> [Integer]
filt _ []    = []
filt n (h:t) = if mod h n /= 0 then h : (filt n t) else filt n t

primes :: [Integer]
primes = primesFrom 2 where primesFrom n = n : filt n (primesFrom (n+1))

-- One recursive call for each element in the first argument
append :: [a] -> [a] -> [a]
append []    xs = xs
append (h:t) xs = h : append t xs

quickSort :: Ord a => [a] -> [a]
quickSort []  = []
quickSort [x] = [x]
quickSort xs  = (quickSort lo) ++ [pv] ++ (quickSort hi)
    where (lo, pv, hi) = partition (randomPivot xs) xs
          randomPivot xs = randomRIO (0, length xs - 1)

partition :: Ord a => Int -> [a] -> ([a], a, [a])
partition _ [] = error "Cannot partition empty list"
partition n xs = (lo, pv, hi)
    where lo = filter (<= pv) noPv
          hi = filter (>  pv) noPv
          pv = xs !! n
          noPv = (take n xs) ++ (drop (n+1) xs)
