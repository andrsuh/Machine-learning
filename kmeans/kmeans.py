class Cluster:
    def __init__(self, point):
        self.center = point.coords
        self.label = point.label
        self.points = {point}
        self.changed = False

    def add_point(self, point):
        self.points.add(point)
        self.changed = True

    def remove_point(self, point):
        self.points.remove(point)
        self.changed = True

    def recalculate(self):
        if not self.changed or not self.points:
            self.changed = False
            return

        dimension = len(self.points)
        points = [point.coords for point in self.points]
        self.center = np.zeros(len(self.center))
        for j in range(len(points[0])):
            self.center[j] = sum([points[i][j]
                                  for i in range(len(points))]) / dimension
        self.changed = False


class Point:
    def __init__(self, coords, label, cluster=None):
        self.coords = coords
        self.label = label
        self.cluster = cluster

    def set_cluster(self, cluster):
        if self.cluster:
            self.cluster.remove_point(self)
        cluster.add_point(self)
        self.cluster = cluster


class KMeans:
    def __init__(self):
        self.clusters = []

    def fit(self, X, y, centroids_idxs):
        # construct points
        self.points = [Point(coords, label) for coords, label in zip(X, y)]

        # initial clusters building
        for i in centroids_idxs:
            cl = Cluster(self.points[i])
            self.clusters.append(cl)
            self.points[i].set_cluster(cl)

        # fit model
        while (True):
            for point in self.points:
                cl = self.find_closest_cluster(point, point.label)
                if cl and cl != point.cluster:
                    point.set_cluster(cl)

            changed = False
            for cl in self.clusters:
                changed |= cl.changed
                cl.recalculate()

            if not changed:
                break

        centroids, labels = [], []
        for cl in self.clusters:
            labels.append(cl.label)
            centroids.append(cl.center)

        return centroids, labels

    def find_closest_cluster(self, point, label):
        cl = min([(cluster, distance(cluster.center, point.coords))
                  for cluster in self.clusters], key=itemgetter(1))[0]
        if cl.label == label:
            return cl
