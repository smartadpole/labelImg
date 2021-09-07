try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.utils.utils import newIcon, labelValidator

BB = QDialogButtonBox


class ClassDialog(QDialog):

    def __init__(self, parent=None, classDicts=None):
        super(ClassDialog, self).__init__(parent)
        self.classes={}
        self.classDicts=classDicts

        layout = QHBoxLayout()

        self.Checklayout = QVBoxLayout()
        self.QComboBox=QComboBox()
        self.QComboBox.addItems(list(self.classDicts.keys()))

        text=self.QComboBox.itemText(0)
        self.checkboxs = self.classDicts[text].copy()
        for i in range(len(self.classDicts[text])):
            self.checkboxs[i] = QCheckBox(str(self.classDicts[text][i]))
            self.Checklayout.addWidget(self.checkboxs[i])
            
        self.QComboBox.currentIndexChanged.connect(self.ComboBoxIndexChanged)

        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)

        layout.addWidget(self.QComboBox)
        layout.addLayout(self.Checklayout)
        layout.addWidget(bb)
        self.setLayout(layout)

    def validate(self):
        index_class = []
        key=list(self.classDicts.keys())[self.QComboBox.currentIndex()]
        for i in range(len(self.checkboxs)):
            if self.checkboxs[i].isChecked():
                index_class.append(self.classDicts[key][i])
        self.classes[key] = index_class
        self.accept()

    def popUp(self,  move=True):
        if move:
            cursor_pos = QCursor.pos()
            parent_bottomRight = self.parentWidget().geometry()
            max_x = parent_bottomRight.x() + parent_bottomRight.width() - self.sizeHint().width()
            max_y = parent_bottomRight.y() + parent_bottomRight.height() - self.sizeHint().height()
            max_global = self.parentWidget().mapToGlobal(QPoint(max_x, max_y))
            if cursor_pos.x() > max_global.x():
                cursor_pos.setX(max_global.x())
            if cursor_pos.y() > max_global.y():
                cursor_pos.setY(max_global.y())
            self.move(cursor_pos)
        return self.classes if self.exec_() else None

    def ComboBoxIndexChanged(self,index):
        index_class = []
        key=list(self.classDicts.keys())[index-1]
        for i in range(len(self.checkboxs)):
            if self.checkboxs[i].isChecked():
                index_class.append(self.classDicts[key][i])
        self.classes[key] = index_class
        self.statusChanged(index)

    def statusChanged(self,index):
        text=self.QComboBox.itemText(index)
        self.checkboxs = self.classDicts[text].copy()

        for i in range(self.Checklayout.count()):
            self.Checklayout.itemAt(i).widget().deleteLater()

        for i in range(len(self.classDicts[text])):
            self.checkboxs[i] = QCheckBox(str(self.classDicts[text][i]))
            self.Checklayout.addWidget(self.checkboxs[i])



