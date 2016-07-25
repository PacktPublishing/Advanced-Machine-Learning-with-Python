neighbors = []
neighbors.append(T.roll(board, 1, 0))
neighbors.append(T.roll(board, 1, 1))
neighbors.append(T.roll(board, -1, 0))
neighbors.append(T.roll(board, -1, 1))
neighbors.append(T.roll(T.roll(board, 1, 1), 1, 0))
neighbors.append(T.roll(T.roll(board, 1, 1), -1, 0))
neighbors.append(T.roll(T.roll(board, -1, 1), -1, 0))
neighbors.append(T.roll(T.roll(board, -1, 1), 1, 0))
alive_neighbors = sum(neighbors)

born = T.eq(board, 0) * T.eq(alive_neighbors, 3)
survived = T.eq(board, 1) * (T.eq(alive_neighbors, 2) + T.eq(alive_neighbors, 3))
new_board = T.cast(survived + born, 'uint8')
updates = {board: new_board}
f = theano.function([], board, updates=updates)