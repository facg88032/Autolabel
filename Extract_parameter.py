import collections

class Extract():
    def __init__(self):
        pass
    def extract_parameter(boxes,
                          scores,
                          classes,
                          category_index,
                          max_boxes=20,
                          min_score_thresh=.5):
        all_box_and_classname = collections.defaultdict(str)
        for i in range(min(max_boxes, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())

                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                all_box_and_classname[box] = class_name
        return all_box_and_classname
