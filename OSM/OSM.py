import osmium

class CounterHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.num_nodes = 0

    def node(self, n):
        self.num_nodes += 1


if __name__ == '__main__':

    h = CounterHandler()

    h.apply_file("greater-london-latest.osm.pbf")

    print("Number of nodes: %d" % h.num_nodes)