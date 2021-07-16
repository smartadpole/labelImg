import sys
try:
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QComboBox
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import QWidget, QHBoxLayout, QComboBox


class ComboBox(QWidget):
    def __init__(self, parent=None, items=[]):
        super(ComboBox, self).__init__(parent)

        layout = QHBoxLayout()
        self.cb = QComboBox()
        self.items = items
        self.cb.addItems(self.items)
        self.parent=parent

        self.cb.currentIndexChanged.connect(self.parent.comboSelectionChanged)

        layout.addWidget(self.cb)
        self.setLayout(layout)

    def update_items(self, items):
        self.items = ["","person_model"]

        # self.cb.clear()
        self.cb.addItems(self.items)
        if self.cb.currentIndex()>=0:
            self.parent.comboSelectionChanged(index=self.cb.currentIndex())
