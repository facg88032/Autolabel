from lxml import etree

class Xml():
    def __init__(self):
        self.annotation = etree.Element('annotation')



    def create_dir_message(self,annotation,folder,filename,path,database='Unknown'):
        self.folder = etree.SubElement(annotation, "folder")
        self.filename = etree.SubElement(annotation, "filename")
        self.path = etree.SubElement(annotation, "path")
        self.source = etree.SubElement(annotation, "source")
        self.database = etree.SubElement(self.source, "database")
        self.database.text = database
        self.folder.text=folder
        self.filename.text = filename
        self.path.text = path
        return annotation

    def image_size(self,annotation,width,height,deepth=3,segmented=0):
        self.size = etree.SubElement(annotation, "size")
        self.segmented = etree.SubElement(annotation, "segmented")
        self.width = etree.SubElement(self.size, "width")
        self.height = etree.SubElement(self.size, "height")
        self.depth = etree.SubElement(self.size, "depth")
        self.segmented.text =str(segmented)
        self.width.text=str(width)
        self.height.text=str(height)
        self.depth.text=str(deepth)
        return annotation

    def create_object(self,annotation,
                      class_name,
                      xmin,xmax,ymin,ymax,
                      pose='Unspecified',
                      truncated=0,
                      difficult=0
                      ):
        self.object=etree.SubElement(annotation,"object")
        self.name=etree.SubElement(self.object, "name")
        self.pose=etree.SubElement(self.object, "pose")
        self.truncated=etree.SubElement(self.object, "truncated")
        self.difficult=etree.SubElement(self.object, "difficult")
        self.bndbox = etree.SubElement(self.object, "bndbox")
        self.xmin=etree.SubElement(self.bndbox, "xmin")
        self.ymin=etree.SubElement(self.bndbox, "ymin")
        self.xmax=etree.SubElement(self.bndbox, "xmax")
        self.ymax=etree.SubElement(self.bndbox, "ymax")
        self.name.text=class_name
        self.pose.text=pose
        self.truncated.text=str(truncated)
        self.difficult.text=str(difficult)
        self.xmin.text=str(int(round(xmin)))
        self.ymin.text=str(int(round(ymin)))
        self.xmax.text=str(int(round(xmax)))
        self.ymax.text=str(int(round(ymax)))
        return annotation



