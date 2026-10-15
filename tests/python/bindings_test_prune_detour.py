import unittest

import numpy as np

import hnswlib


def level0_lists(index):
    """Level-0 neighbor ids (padded with -1) and the entry point, read from the pickled state."""
    params = index.__getstate__()[0]
    count, size, max_m0 = params['cur_element_count'], params['size_data_per_element'], params['max_M0']
    raw = np.asarray(params['data_level0']).view(np.uint8)[:count * size].reshape(count, size)
    lists = np.ascontiguousarray(raw[:, :4 * (max_m0 + 1)]).view('<u4')
    sizes = (lists[:, 0] & 0xFFFF).astype(np.int64)
    links = lists[:, 1:].astype(np.int64)
    links[np.arange(max_m0)[None, :] >= sizes[:, None]] = -1
    return links, params['enterpoint_node']


def graph_bytes(index):
    params = index.__getstate__()[0]
    return np.asarray(params['data_level0']).tobytes() + np.asarray(params['link_lists']).tobytes()


def unreachable_count(index):
    """Elements that no level-0 path from the entry point reaches."""
    links, entry_point = level0_lists(index)
    seen = np.zeros(len(links), dtype=bool)
    seen[entry_point] = True
    frontier = np.array([entry_point])
    while len(frontier):
        neighbors = np.unique(links[frontier].reshape(-1))
        neighbors = neighbors[neighbors >= 0]
        frontier = neighbors[~seen[neighbors]]
        seen[frontier] = True
    return int((~seen).sum())


def build(data, M, ef_construction, detour, num_threads, link_order='per-level'):
    index = hnswlib.Index(space='l2', dim=data.shape[1])
    index.init_index(max_elements=len(data), M=M, ef_construction=ef_construction, random_seed=5)
    index.set_insert_link_order(link_order)
    if detour is not None:
        index.set_prune_detour_check(detour)
    index.add_items(data, np.arange(len(data)), num_threads=num_threads)
    return index


class PruneDetourCheckTestCase(unittest.TestCase):
    def setUp(self):
        self.data = np.random.default_rng(3).random((2000, 16), dtype=np.float32)

    def testOffByDefaultAndRequiresIndex(self):
        index = hnswlib.Index(space='l2', dim=4)
        with self.assertRaises(RuntimeError):
            index.set_prune_detour_check(True)
        with self.assertRaises(RuntimeError):
            index.set_ef_construction(10)
        index.init_index(max_elements=10, ef_construction=20, M=4)
        self.assertFalse(index.get_prune_detour_check())
        index.set_prune_detour_check(True)
        self.assertTrue(index.get_prune_detour_check())
        self.assertEqual(set(hnswlib.Index.get_prune_detour_stats()),
                         {'prunes', 'prunes_with_drops', 'dropped_old', 'verified', 'refilled', 'swapped',
                          'left_unchanged', 'rescue_needed', 'rescue_linked', 'rescue_failed', 'witness_busy'})

    def testSetEfConstructionMatchesInitIndex(self):
        index = hnswlib.Index(space='l2', dim=16)
        index.init_index(max_elements=len(self.data), M=8, ef_construction=200, random_seed=5)
        index.set_ef_construction(3)
        self.assertEqual(index.ef_construction, 8)  # at least M
        index.set_ef_construction(12)
        self.assertEqual(index.ef_construction, 12)
        index.add_items(self.data, np.arange(len(self.data)), num_threads=1)
        self.assertEqual(graph_bytes(index), graph_bytes(build(self.data, 8, 12, None, 1)))

    def testFlagOffKeepsTheDefaultGraph(self):
        # Single-threaded builds are deterministic; turning the check off must leave the build as if
        # it had never been set, and must not count anything.
        hnswlib.Index.reset_prune_detour_stats()
        for link_order in ('per-level', 'own-lists-first'):
            default = build(self.data, 8, 8, None, 1, link_order)
            index = hnswlib.Index(space='l2', dim=16)
            index.init_index(max_elements=len(self.data), M=8, ef_construction=8, random_seed=5)
            index.set_insert_link_order(link_order)
            index.set_prune_detour_check(True)
            index.set_prune_detour_check(False)
            index.add_items(self.data, np.arange(len(self.data)), num_threads=1)
            self.assertEqual(graph_bytes(index), graph_bytes(default))
        self.assertEqual(sum(hnswlib.Index.get_prune_detour_stats().values()), 0)

    def testDetourCheckKeepsEveryElementReachable(self):
        for num_threads in (1, 4):
            for link_order in ('per-level', 'own-lists-first'):
                hnswlib.Index.reset_prune_detour_stats()
                index = build(self.data, 8, 8, True, num_threads, link_order)
                stats = hnswlib.Index.get_prune_detour_stats()
                case = (num_threads, link_order, stats)
                self.assertEqual(unreachable_count(index), 0, case)
                self.assertGreater(stats['prunes'], 0, case)
                self.assertGreater(stats['dropped_old'], stats['verified'], case)
                self.assertGreater(stats['verified'], 0, case)
                self.assertEqual(stats['rescue_failed'], 0, case)
                self.assertEqual(stats['rescue_needed'], stats['rescue_linked'], case)

                links, _ = level0_lists(index)
                for element, row in enumerate(links):
                    row = row[row >= 0]
                    self.assertEqual(len(set(row.tolist())), len(row))
                    self.assertNotIn(element, row)

                index.set_ef(50)
                labels, _ = index.knn_query(self.data[:200], k=1)
                self.assertGreater(np.mean(labels[:, 0] == np.arange(200)), 0.9)

    def testBaselineLosesElementsWhereTheCheckKeepsThem(self):
        # M=4 prunes hard enough that the normal heuristic cuts some elements off.
        self.assertGreater(unreachable_count(build(self.data, 4, 4, False, 1)), 0)
        self.assertEqual(unreachable_count(build(self.data, 4, 4, True, 1)), 0)


if __name__ == '__main__':
    unittest.main()
