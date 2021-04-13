
class CenterNet:
    def init(self, file):
        if os.path.isfile(file):
            self.net = ONNXModel(file)
        else:
            raise IOError("no such file {}".format(self.model_file_4_detect))

    def forward():
        img_input, meta = pre_process(gray, IMAGE_SIZE_CENTER_NET, 1, None)
        timer.Timing("preprocess")

        out = net.forward(img_input)
        timer.Timing("inference")
        print()
        results_batch = PostProcessor_CENTER_NET(out, meta)

        for result in results_batch.values():
            if len(result) > 0:
                result = result.tolist()
                result = [r for r in result if r[SCORE_ID] > CONFIDENCE_THRESHOLD]
                for r in result:
                    x, y, x2, y2, score = r
                    x, y, x2, y2, score = int(x), int(y), int(x2), int(y2), float(score)