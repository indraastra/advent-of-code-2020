from collections import defaultdict
from dataclasses import dataclass, field
import heapq
from typing import Any, Set


def neighboring_cells(cell):
    x, y = cell
    return ((x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1))


@dataclass(frozen=True, eq=True)
class Tile:
    tile_id: int
    top: str
    bottom: str
    left: str
    right: str
    flipped: bool = False
    rotation: int = 0

    @staticmethod
    def from_record(record):
        tile_id, tile_map = record.split(':')
        tile_id = int(tile_id.strip().split()[-1])
        rows = tile_map.strip().splitlines()
        top = rows[0]
        bottom = rows[-1]
        left = ''.join(row[0] for row in rows)
        right = ''.join(row[-1] for row in rows)
        return Tile(tile_id, top, bottom, left, right)

    def edges(self):
        return set([self.top, self.bottom, self.left, self.right])

    def flip(self):
        return Tile(self.tile_id,
                    self.top[::-1], self.bottom[::-1], self.right, self.left,
                    not self.flipped, self.rotation)

    def rotate(self, amount):
        rotation = (self.rotation + amount) % 360
        if amount == 90:
            return Tile(self.tile_id, self.left[::-1], self.right[::-1], self.bottom, self.top, self.flipped, rotation)
        elif amount == 180:
            return Tile(self.tile_id, self.bottom[::-1], self.top[::-1], self.right[::-1], self.left[::-1], self.flipped, rotation)
        elif amount == 270:
            return Tile(self.tile_id, self.right, self.left, self.top[::-1], self.bottom[::-1], self.flipped, rotation)
        else:
            return self

    def orientations(self):
        for deg in [0, 90, 180, 270]:
            yield self.rotate(deg)
            yield self.flip().rotate(deg)

    def tile_key(self):
        return (self.tile_id, self.flipped, self.rotation)

    def fits_left_of(self, tile):
        return self.right == tile.left

    def fits_above(self, tile):
        return self.bottom == tile.top

    def __hash__(self):
        return hash(self.tile_key())


class TileMap:
    def __init__(self, tiles=None, cells=None):
        self.tiles = tiles or frozenset()
        self.cells = cells or {}

    def available_cells(self):
        if not self.cells:
            yield (0, 0)
        else:
            for cell in self.cells:
                for neighbor in neighboring_cells(cell):
                    if not self.cells.get(neighbor):
                        yield neighbor

    def neighboring_tiles(self, cell):
        return [self.cells.get(neighbor) for neighbor in neighboring_cells(cell)]

    def fits(self, tile, cell):
        neighbors = self.neighboring_tiles(cell)
        left_tile, above_tile, right_tile, below_tile = neighbors
        return ((not left_tile or left_tile.fits_left_of(tile)) and
                (not right_tile or tile.fits_left_of(right_tile)) and
                (not above_tile or above_tile.fits_above(tile)) and
                (not below_tile or tile.fits_above(below_tile)))

    def place(self, tile, cell):
        new_cells = dict(self.cells)
        new_cells[cell] = tile
        return TileMap(self.tiles | {tile}, new_cells)

    def placements(self, tile):
        orientations = list(tile.orientations()) if self.tiles else [tile]
        for cell in self.available_cells():
            for tile in orientations:
                if self.fits(tile, cell):
                    yield tile, cell

    def bounds(self):
        x_lo = min(self.cells, key=lambda xy: xy[0])[0]
        x_hi = max(self.cells, key=lambda xy: xy[0])[0]
        y_lo = min(self.cells, key=lambda xy: xy[1])[1]
        y_hi = max(self.cells, key=lambda xy: xy[1])[1]
        return x_lo, x_hi, y_lo, y_hi

    def size(self):
        x_lo, x_hi, y_lo, y_hi = self.bounds()
        return x_hi - x_lo + 1, y_hi - y_lo + 1

    def is_filled(self):
        x_lo, x_hi, y_lo, y_hi = self.bounds()
        for x in range(x_lo, x_hi+1):
            for y in range(y_lo, y_hi+1):
                if not self.cells.get((x, y)):
                    return False
        return True

    def corners(self):
        x_lo, x_hi, y_lo, y_hi = self.bounds()
        return [self.cells[(x_lo, y_lo)],
                self.cells[(x_hi, y_lo)],
                self.cells[(x_lo, y_hi)],
                self.cells[(x_hi, y_hi)]]

    def print(self):
        x_lo, x_hi, y_lo, y_hi = self.bounds()
        for y in range(y_lo, y_hi+1):
            for x in range(x_lo, x_hi+1):
                print(
                    f'{self.cells[(x,y)].tile_id if (x,y) in self.cells else ".":<6}', end='')
            print()

    def __lt__(self, other):
        return len(self.tiles) < len(other.tiles)


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


def place_all(tiles, size):
    placements = []
    tile_map = TileMap()

    def add_placement(tile_map, tile, cell, remaining):
        neighbors = sum(
            t is not None for t in tile_map.neighboring_tiles(cell))
        priority = (len(remaining), -neighbors)
        heapq.heappush(placements, PrioritizedItem(
            priority, (tile_map, tile, cell, remaining)))

    def get_placement():
        return heapq.heappop(placements).item

    for tile, cell in tile_map.placements(tiles[0]):
        remaining = frozenset(tiles[1:])
        add_placement(tile_map, tile, cell, remaining)

    while placements:
        tile_map, tile, cell, remaining = get_placement()
        new_tile_map = tile_map.place(tile, cell)
        if any(dim > size for dim in new_tile_map.size()):
            continue
        if not remaining and new_tile_map.is_filled():
            return new_tile_map
        for new_tile in remaining:
            new_remaining = remaining - {new_tile}
            for new_tile, new_cell in new_tile_map.placements(new_tile):
                add_placement(new_tile_map, new_tile, new_cell, new_remaining)


# Archives.


@dataclass(frozen=True, eq=True)
class Edge:
    position: str
    tile1: Tile
    tile2: Tile


@dataclass(eq=True, order=True)
class TileMapV1:
    tiles: Set[Tile] = field(default_factory=set)
    edges: Set[Edge] = field(default_factory=set)

    def try_placing(self, tile):
        if not self.tiles:
            yield TileMapV1(self.tiles | {tile}, self.edges)

        for other_tile in self.tiles:
            for oriented in tile.orientations():
                if oriented.fits_left_of(other_tile):
                    yield self.place_left_of(oriented, other_tile)
                if other_tile.fits_left_of(oriented):
                    yield self.place_left_of(other_tile, oriented)
                if oriented.fits_above(other_tile):
                    yield self.place_above(oriented, other_tile)
                if other_tile.fits_above(oriented):
                    yield self.place_above(other_tile, oriented)

    def place_left_of(self, t1, t2):
        return TileMapV1(self.tiles | {t1, t2},
                         self.edges | {Edge('left', t1, t2)})

    def place_above(self, t1, t2):
        return TileMapV1(self.tiles | {t1, t2},
                         self.edges | {Edge('above', t1, t2)})

    def complete_edges(self):
        edge_completions = defaultdict(lambda: None)
        all_edges = set()

        def add_edge(edge):
            if edge in all_edges:
                return False
            all_edges.add(edge)
            edge_completions[(edge.tile1, edge.position)] = edge.tile2
            edge_completions[(edge.position, edge.tile2)] = edge.tile1
            return False

        for edge in self.edges:
            add_edge(edge)

        while True:
            changed = False
            for edge in list(all_edges):
                changed |= add_edge(edge)
                position, t1, t2 = edge.position, edge.tile1, edge.tile2
                new_edges = []
                if edge.position == 'left':
                    new_edges.append(
                        ('left', edge_completions[(t1, 'above')],  edge_completions[(t2, 'above')]))
                    new_edges.append(
                        ('left', edge_completions[('above', t1)], edge_completions[('above', t2)]))
                elif edge.position == 'above':
                    new_edges.append(
                        ('above', edge_completions[(t1, 'left')],  edge_completions[(t2, 'left')]))
                    new_edges.append(
                        ('above', edge_completions[('left', t1)], edge_completions[('left', t2)]))
                for position, t1, t2 in new_edges:
                    if t1 and t2:
                        changed |= add_edge(Edge(position, t1, t2))
            if not changed:
                break
        return all_edges, edge_completions

    def is_solved(self, m, n):
        all_edges, _ = self.complete_edges()
        return len(all_edges) == (m * (m - 1) + n * (n - 1))

    def corners(self):
        _, edge_completions = self.complete_edges()
        tl = tr = bl = br = None
        for t in self.tiles:
            if not edge_completions[('above', t)] and not edge_completions[('left', t)]:
                tl = t
            elif not edge_completions[('above', t)] and not edge_completions[(t, 'left')]:
                tr = t
            elif not edge_completions[(t, 'above')] and not edge_completions[('left', t)]:
                bl = t
            elif not edge_completions[(t, 'above')] and not edge_completions[(t, 'left')]:
                br = t
        return tl, tr, bl, br


def place_all_v1(tiles, size):
    possibilities = []

    def add_possibility(tile_map, remaining):
        heapq.heappush(possibilities, (len(remaining),
                                       tile_map, frozenset(remaining)))

    initial_tile_map = list(TileMapV1().try_placing(tiles[0]))[0]
    add_possibility(initial_tile_map, tiles[1:])

    while possibilities:
        _, tile_map, remaining = heapq.heappop(possibilities)
        if not remaining and tile_map.is_solved(size, size):
            return tile_map
        for tile in remaining:
            new_remaining = remaining - {tile}
            for new_tile_map in tile_map.try_placing(tile):
                add_possibility(new_tile_map, new_remaining)


def place_one_v2(tile_map, tile):
    orientations = list(tile.orientations()) if tile_map.tiles else [tile]
    for cell in tile_map.available_cells():
        for tile in orientations:
            if tile_map.fits(tile, cell):
                yield cell, tile_map.place(tile, cell)


def place_all_v2(tiles, size):
    possibilities = []

    def add_possibility(tile_map, remaining, cell):
        p1 = len(remaining)
        p2 = -sum(t is not None for t in tile_map.neighboring_tiles(cell))
        heapq.heappush(possibilities, ((p1, p2), tile_map, remaining))

    def get_possibility():
        return heapq.heappop(possibilities)[1:]

    for cell, tile_map in place_one_v2(TileMap(), tiles[0]):
        add_possibility(tile_map, frozenset(tiles[1:]), cell)

    while possibilities:
        tile_map, remaining = get_possibility()
        if not remaining and tile_map.is_filled():
            return tile_map
        for tile in remaining:
            new_remaining = remaining - {tile}
            for cell, new_tile_map in place_one_v2(tile_map, tile):
                if all(dim <= size for dim in new_tile_map.size()):
                    add_possibility(new_tile_map, new_remaining, cell)
