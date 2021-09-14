from matplotlib import pyplot as plt
from matplotlib import patches

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    
    bottom_left = (xmin, ymin)
    width = xmax - xmin
    height = ymax - ymin
    
    return bottom_left, width, height

def draw_pascal_voc_bboxes(
    plot_ax, bboxes, labels,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for index, bbox in enumerate(bboxes):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)
        rect_1  = patches.Rectangle(
            bottom_left, width, height, linewidth=4,
            edgecolor='black', fill=False
        )
        
        rect_2  = patches.Rectangle(
            bottom_left, width, height, linewidth=2,
            edgecolor='white', fill=False
        )
        
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)
        plot_ax.text(bottom_left[0], bottom_left[1], labels[index], fontsize=15, color='white')
        

def show_image(
    image, bboxes=None, class_labels=None,
    draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes, class_labels)
        
    plt.show()