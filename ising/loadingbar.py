"""Quick module I wrote for pretty textual loading bars!"""

from warnings import warn


class LoadingBar:

    def __init__(self, iterations, displaysize=100):

        self.iterations = iterations
        self.size = displaysize

        self._itercount = 0
        self._progress = 0
        self.running = False

        self.interruptmsg = "/interrupt"
        self.progressbarchar = "."
        self.initbarchar = "="
        self.edges = ("| ", " |")


    def print_init(self):

        print(self.edges[0] + (self.size * self.initbarchar) + self.edges[1])
        print(end=self.edges[0])
        self.running = True

    def print_next(self):

        if self.running:

            self._itercount += 1

            next_progress = int(self.size * self._itercount / self.iterations)

            print(end=(next_progress - self._progress) * self.progressbarchar)

            self._progress = next_progress

            if self._itercount >= self.iterations:
                self.reset(completed=True)

        else:

            self.print_init()
            self.print_next()

    def interrupt(self):
        self.reset(completed=False)

    def reset(self, completed=False):

        if completed:
            print(self.edges[1])
        else:
            if self.running:
                print(self.interruptmsg)

        self._progress = 0
        self._itercount = 0
        self.running = False
