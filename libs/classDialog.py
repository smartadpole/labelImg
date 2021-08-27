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

    def __init__(self, text="Enter object label", parent=None, listItem=None):
        super(ClassDialog, self).__init__(parent)

        layout = QVBoxLayout()

        self.checkboxs=listItem.copy()
        for i in range(len(listItem)):
            self.checkboxs[i] = QCheckBox(str(listItem[i]))
            layout.addWidget(self.checkboxs[i])
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self.setLayout(layout)

    def validate(self):
        self.accept()
        # try:
        #     if self.edit.text().trimmed():
        #         self.accept()
        # except AttributeError:
        #     # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
        #     if self.edit.text().strip():
        #         self.accept()

    def popUp(self,  move=True):
        classes = []
        for i in range(len(self.checkboxs)):
            if self.checkboxs[i].isChecked():
                classes.append(self.datasets[i])
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
        return classes if self.exec_() else None


