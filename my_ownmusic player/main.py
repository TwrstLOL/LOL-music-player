import sys
import os
import random
import time
import math
import numpy as np
from pathlib import Path

# --------------------------
# Dependencies Check
# --------------------------
try:
    import pygame
    from PIL import Image, ImageStat
except ImportError as e:
    raise RuntimeError(f"Missing dependency: {e}. Install with: pip install pygame pillow numpy")

# Mutagen for metadata
try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QListWidget, QSlider, QMessageBox, QFrame,
    QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect, QLineEdit, QMenu, QAbstractItemView,
    QDialog, QListWidgetItem, QProgressDialog, QInputDialog
)
from PyQt6.QtGui import (
    QPainter, QColor, QFont, QPixmap, QAction, QLinearGradient, 
    QBrush, QPen, QPainterPath, QRadialGradient
)
from PyQt6.QtGui import QIcon, QKeySequence
from PyQt6.QtCore import Qt, QTimer, QPoint, QRectF, QSize, pyqtSignal, QObject, QPropertyAnimation
import threading

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# Initialize pygame mixer
pygame.mixer.init()

def basename(path):
    return os.path.basename(path)

# --------------------------
# Color Extraction Logic
# --------------------------
def get_dominant_colors(pil_image, num_colors=2):
    """
    Extracts dominant colors from a PIL image using simple quantization 
    to avoid heavy sklearn dependency.
    """
    try:
        # Resize to speed up processing
        img = pil_image.copy()
        img.thumbnail((150, 150))
        
        # Palettize to reduce colors
        result = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        result.putalpha(0)
        colors = result.getcolors(150*150)
        
        # Sort by count
        ordered = sorted(colors, key=lambda x: x[0], reverse=True)
        
        # Extract RGB values
        rgb_colors = []
        palette = result.getpalette()
        for count, index in ordered[:num_colors]:
            start = index * 3
            rgb = palette[start:start+3]
            rgb_colors.append(QColor(rgb[0], rgb[1], rgb[2]))
            
        return rgb_colors
    except Exception as e:
        print(f"Color extract error: {e}")
        return [QColor(20, 20, 20), QColor(40, 0, 60)]

# --------------------------
# Advanced Visualizer
# --------------------------
class AdvancedVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        # Using 128 bars total, but we will mirror them (64 left, 64 right)
        self.bars = 128
        self.values = np.zeros(self.bars)
        self.peaks = np.zeros(self.bars)
        self.decay = np.ones(self.bars) * 0.05
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16) # ~60 FPS
        
        self.active = False
        self.phase_offset = 0.0

    def _tick(self):
        if not self.active:
            # Idle gentle wave
            t = time.time()
            self.values = 0.05 + 0.03 * np.sin(np.arange(self.bars) * 0.2 + t * 2)
            self.peaks = np.maximum(self.peaks, self.values)
            self.peaks -= 0.005 # Slow decay for peaks in idle
            self.update()
            return

        # Simulation Logic (mimicking FFT)
        # We generate noise but smooth it to look like spectrum data
        self.phase_offset += 0.2
        
        # Generate raw noise
        noise = np.random.rand(self.bars)
        
        # Smooth noise (simulating frequency bands coherence)
        # Simple moving average
        smoothed = np.convolve(noise, np.ones(3)/3, mode='same')
        
        # Apply a "beat" multiplier based on time to simulate kicks
        beat = 1.0 + 0.5 * math.pow(math.sin(time.time() * 6), 20) 
        
        # Frequency shape: Higher energy in bass (left) usually, but for mirror we want center bass
        # Let's generate a full spectrum and then we will split it in paintEvent
        spectrum = smoothed * beat * 0.8
        
        # Smooth transition from previous frame (Linear Interpolation)
        self.values = self.values * 0.7 + spectrum * 0.3
        
        # Peak logic
        mask = self.values > self.peaks
        self.peaks[mask] = self.values[mask]
        self.peaks[~mask] -= 0.01  # Gravity fall
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        
        # We draw from center out
        center_x = w / 2
        bar_w = (w / self.bars) * 0.8
        spacing = (w / self.bars) * 0.2
        
        # Gradient for bars
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(255, 255, 255, 100))
        grad.setColorAt(1.0, QColor(0, 255, 200, 240))
        
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)

        # Peak pen
        peak_pen = QPen(QColor(255, 100, 100, 200))
        peak_pen.setWidth(2)

        # Mirror Logic:
        # We actually calculate 32 bars, then draw them left and right of center
        half_bars = self.bars // 2

        for i in range(half_bars):
            # Bass is usually at index 0, highs at index 32
            # We want Bass in the center
            val_idx = i 
            
            val = max(0.02, self.values[val_idx])
            peak = max(0.02, self.peaks[val_idx])
            
            bar_h = val * h * 0.8
            peak_y = h - (peak * h * 0.8)
            
            # Left side
            x_left = center_x - ((i + 1) * (bar_w + spacing))
            # Right side
            x_right = center_x + (i * (bar_w + spacing))
            
            # Draw Bars
            # Rounded caps look more "iOS"
            path_l = QPainterPath()
            path_l.addRoundedRect(x_left, h - bar_h, bar_w, bar_h, 4, 4)
            painter.drawPath(path_l)
            
            path_r = QPainterPath()
            path_r.addRoundedRect(x_right, h - bar_h, bar_w, bar_h, 4, 4)
            painter.drawPath(path_r)
            
            # Draw Peaks (floating dashes)
            painter.setPen(peak_pen)
            painter.drawLine(int(x_left), int(peak_y), int(x_left + bar_w), int(peak_y))
            painter.drawLine(int(x_right), int(peak_y), int(x_right + bar_w), int(peak_y))
            
            painter.setPen(Qt.PenStyle.NoPen)

class MarketplaceDialog(QDialog):
    """Simple marketplace dialog showing a few sample free tracks and allowing download.
    Downloads are performed in a background thread and can be imported into the main playlist.
    """
    def __init__(self, parent=None, on_download=None):
        super().__init__(parent)
        self.on_download = on_download
        self.setWindowTitle("Marketplace")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        self.store_list = QListWidget()
        layout.addWidget(self.store_list)

        btn_row = QHBoxLayout()
        self.btn_download = QPushButton("Download")
        self.btn_import_url = QPushButton("Import from URL")
        self.btn_close = QPushButton("Close")
        btn_row.addWidget(self.btn_download)
        btn_row.addWidget(self.btn_import_url)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        # Sample items (public sample MP3s). Only include samples meant for testing.
        samples = [
            {"title": "Sample 1", "artist": "SoundHelix", "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"},
            {"title": "Sample 2", "artist": "SoundHelix", "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"},
            {"title": "Sample 3", "artist": "SoundHelix", "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"},
        ]

        for s in samples:
            it = QListWidgetItem(f"{s['title']} — {s['artist']}")
            it.setData(Qt.ItemDataRole.UserRole, s)
            self.store_list.addItem(it)

        self.btn_download.clicked.connect(self._download_selected)
        self.btn_import_url.clicked.connect(self._import_from_url)
        self.btn_close.clicked.connect(self.accept)

        # Signals object for progress updates
        class _DLSignals(QObject):
            progress = pyqtSignal(int)
            finished = pyqtSignal(str)
            error = pyqtSignal(str)

        self._signals = _DLSignals()

    def _download_selected(self):
        if not REQUESTS_AVAILABLE:
            QMessageBox.warning(self, "Requests missing", "The 'requests' package is required to download. Install it with: pip install requests")
            return

        it = self.store_list.currentItem()
        if not it:
            QMessageBox.information(self, "No selection", "Please select a track to download.")
            return
        meta = it.data(Qt.ItemDataRole.UserRole)
        url = meta.get('url')
        suggested = meta.get('title', 'download') + Path(url).suffix
        dest, _ = QFileDialog.getSaveFileName(self, "Save track as", suggested, "Audio (*.mp3 *.wav *.ogg)")
        if not dest:
            return

        progress = QProgressDialog("Downloading…", "Cancel", 0, 100, self)
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        def on_progress(val):
            progress.setValue(val)

        def on_finished(path):
            progress.setValue(100)
            QMessageBox.information(self, "Downloaded", f"Saved to: {path}")
            if callable(self.on_download):
                self.on_download(path, True)

        def on_error(msg):
            QMessageBox.critical(self, "Download error", msg)

        self._signals.progress.connect(on_progress)
        self._signals.finished.connect(on_finished)
        self._signals.error.connect(on_error)

        def worker():
            try:
                r = requests.get(url, stream=True, timeout=15)
                r.raise_for_status()
                total = r.headers.get('content-length')
                if total is None:
                    # unknown size
                    with open(dest, 'wb') as fh:
                        fh.write(r.content)
                    self._signals.progress.emit(100)
                    self._signals.finished.emit(dest)
                    return

                total = int(total)
                written = 0
                chunk_size = 8192
                with open(dest, 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        pct = int((written / total) * 100)
                        self._signals.progress.emit(pct)
                self._signals.finished.emit(dest)
            except Exception as e:
                self._signals.error.emit(str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _import_from_url(self):
        if not REQUESTS_AVAILABLE:
            QMessageBox.warning(self, "Requests missing", "The 'requests' package is required to download. Install it with: pip install requests")
            return

        url, ok = QInputDialog.getText(self, "Import from URL", "Enter direct audio file URL:")
        if not ok or not url:
            return
        # Ask where to save
        suggested = Path(url).name or 'download.mp3'
        dest, _ = QFileDialog.getSaveFileName(self, "Save track as", suggested, "Audio (*.mp3 *.wav *.ogg)")
        if not dest:
            return
        # Reuse download logic by adding a temporary item and invoking worker
        # Show simple progress dialog
        progress = QProgressDialog("Downloading…", "Cancel", 0, 0, self)
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        def worker():
            try:
                r = requests.get(url, stream=True, timeout=15)
                r.raise_for_status()
                with open(dest, 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                self._signals.finished.emit(dest)
            except Exception as e:
                self._signals.error.emit(str(e))

        def on_finished(path):
            progress.close()
            QMessageBox.information(self, "Downloaded", f"Saved to: {path}")
            if callable(self.on_download):
                self.on_download(path, True)

        def on_error(msg):
            progress.close()
            QMessageBox.critical(self, "Download error", msg)

        self._signals.finished.connect(on_finished)
        self._signals.error.connect(on_error)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        
        # Peak pen
        peak_pen = QPen(QColor(255, 100, 100, 200))
        peak_pen.setWidth(2)
        
        # Mirror Logic:
        # We actually calculate 32 bars, then draw them left and right of center
        half_bars = self.bars // 2
        
        for i in range(half_bars):
            # Bass is usually at index 0, highs at index 32
            # We want Bass in the center
            val_idx = i 
            
            val = max(0.02, self.values[val_idx])
            peak = max(0.02, self.peaks[val_idx])
            
            bar_h = val * h * 0.8
            peak_y = h - (peak * h * 0.8)
            
            # Left side
            x_left = center_x - ((i + 1) * (bar_w + spacing))
            # Right side
            x_right = center_x + (i * (bar_w + spacing))
            
            # Draw Bars
            # Rounded caps look more "iOS"
            path_l = QPainterPath()
            path_l.addRoundedRect(x_left, h - bar_h, bar_w, bar_h, 4, 4)
            painter.drawPath(path_l)
            
            path_r = QPainterPath()
            path_r.addRoundedRect(x_right, h - bar_h, bar_w, bar_h, 4, 4)
            painter.drawPath(path_r)
            
            # Draw Peaks (floating dashes)
            painter.setPen(peak_pen)
            painter.drawLine(int(x_left), int(peak_y), int(x_left + bar_w), int(peak_y))
            painter.drawLine(int(x_right), int(peak_y), int(x_right + bar_w), int(peak_y))
            
            painter.setPen(Qt.PenStyle.NoPen)

# --------------------------
# Main Player Window
# --------------------------
class NeonPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Concept Player")
        self.resize(1100, 750)
        
        # Reactive Background State
        self.bg_color_1 = QColor(10, 10, 20)
        self.bg_color_2 = QColor(30, 0, 40)
        self.target_color_1 = QColor(10, 10, 20)
        self.target_color_2 = QColor(30, 0, 40)
        self.bg_timer = QTimer(self)
        self.bg_timer.timeout.connect(self._animate_background)
        self.bg_timer.start(20) # Smooth transition
        
        # Styling
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) 
        # We handle background in paintEvent manually for the gradient
        
        self.setStyleSheet("""
            QWidget { 
                font-family: 'Segoe UI', sans-serif; 
                color: white; 
            }
            QListWidget { 
                background-color: rgba(0, 0, 0, 0.3); 
                border-radius: 15px; 
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #ddd;
                font-size: 14px;
                padding: 10px;
            }
            QListWidget::item:selected {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
            QPushButton { 
                background-color: rgba(255, 255, 255, 0.1); 
                border: none;
                border-radius: 20px; 
                padding: 10px; 
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: rgba(255, 255, 255, 0.2); 
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.05);
            }
            QLabel { color: rgba(255, 255, 255, 0.9); }
            QSlider::groove:horizontal { 
                height: 6px; 
                background: rgba(255, 255, 255, 0.2); 
                border-radius: 3px; 
            }
            QSlider::handle:horizontal { 
                background: white; 
                width: 16px; 
                height: 16px; 
                margin: -5px 0; 
                border-radius: 8px; 
            }
        """)

        # --- Data ---
        self.playlist = []
        self.current_index = -1
        self.playing = False
        self.loop = False
        self.shuffle = False
        
        # --- UI Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # 1. Top Section: Album Art & Info
        top_layout = QHBoxLayout()
        
        # Album Art (Left)
        self.album_art = QLabel()
        self.album_art.setFixedSize(300, 300)
        self.album_art.setStyleSheet("""
            background-color: rgba(0,0,0,0.5); 
            border-radius: 20px; 
            border: 1px solid rgba(255,255,255,0.1);
        """)
        self.album_art.setScaledContents(True)
        # Shadow for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(50)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 10)
        self.album_art.setGraphicsEffect(shadow)
        
        # Track Info & Visualizer (Right)
        info_layout = QVBoxLayout()
        self.title_label = QLabel("Welcome")
        self.title_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.artist_label = QLabel("Load tracks to begin")
        self.artist_label.setFont(QFont("Segoe UI", 16))
        self.artist_label.setStyleSheet("color: rgba(255,255,255,0.6);")
        
        self.visualizer = AdvancedVisualizer()
        
        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.artist_label)
        info_layout.addSpacing(20)
        info_layout.addWidget(self.visualizer)
        info_layout.addStretch()
        
        top_layout.addWidget(self.album_art)
        top_layout.addSpacing(40)
        top_layout.addLayout(info_layout, stretch=1)
        
        main_layout.addLayout(top_layout)
        
        # 2. Middle: Controls
        # We put controls in a "Glass" container
        controls_container = QFrame()
        controls_container.setStyleSheet(".QFrame { background: rgba(0,0,0,0.2); border-radius: 25px; }")
        controls_layout = QVBoxLayout(controls_container)
        
        # Progress (Fake logic for this demo, would need MP3 length polling)
        # Playback progress: elapsed / slider / total
        progress_row = QHBoxLayout()
        self.label_elapsed = QLabel("0:00")
        self.label_elapsed.setFixedWidth(50)
        progress_row.addWidget(self.label_elapsed)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setValue(0)
        self.progress_slider.setSingleStep(1)
        progress_row.addWidget(self.progress_slider, stretch=1)

        self.label_total = QLabel("0:00")
        self.label_total.setFixedWidth(50)
        progress_row.addWidget(self.label_total)

        controls_layout.addLayout(progress_row)

        # Internal state for tracking playback position and length
        self.current_track_length = 0.0
        self._current_play_offset = 0.0
        self._seeking = False
        
        btns_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⏮")
        self.btn_play = QPushButton("▶ Play")
        self.btn_next = QPushButton("⏭")
        self.btn_loop = QPushButton("repeat")
        self.btn_shuffle = QPushButton("SHUFFLE")
        
        self.btn_play.setFixedSize(120, 50)
        self.btn_play.setStyleSheet("background-color: white; color: black; font-size: 16px;")
        
        for b in [self.btn_loop, self.btn_prev, self.btn_play, self.btn_next, self.btn_shuffle]:
            btns_layout.addWidget(b)
            
        controls_layout.addLayout(btns_layout)
        # Crossfade control
        xf_layout = QHBoxLayout()
        xf_layout.addWidget(QLabel("Crossfade (ms)"))
        self.crossfade_slider = QSlider(Qt.Orientation.Horizontal)
        self.crossfade_slider.setRange(0, 5000)
        self.crossfade_slider.setValue(0)
        self.crossfade_slider.setFixedWidth(200)
        xf_layout.addWidget(self.crossfade_slider)
        controls_layout.addLayout(xf_layout)
        main_layout.addSpacing(20)
        main_layout.addWidget(controls_container)

        # 3. Bottom: Playlist & File Ops
        bottom_layout = QHBoxLayout()
        # Playlist column (with header and search)
        playlist_col = QVBoxLayout()

        header_row = QHBoxLayout()
        self.playlist_label = QLabel("Tracks (0)")
        self.playlist_label.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        header_row.addWidget(self.playlist_label)
        header_row.addStretch()
        self.btn_pin_playlist = QPushButton("📌")
        self.btn_pin_playlist.setCheckable(True)
        self.btn_pin_playlist.setToolTip("Pin playlist expanded")
        header_row.addWidget(self.btn_pin_playlist)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter tracks…")
        self.search_box.setFixedHeight(28)

        self.playlist_widget = QListWidget()
        # Allow drag-drop reordering
        self.playlist_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.playlist_widget.setAlternatingRowColors(True)
        self.playlist_widget.doubleClicked.connect(self._on_item_double_click)
        
        ops_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load Files")
        self.btn_fullscreen = QPushButton("⛶ Fullscreen")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedWidth(150)
        
        ops_layout.addWidget(self.btn_load)
        self.btn_add_folder = QPushButton("Add Folder")
        ops_layout.addWidget(self.btn_add_folder)
        ops_layout.addWidget(self.btn_fullscreen)
        self.btn_market = QPushButton("Marketplace")
        ops_layout.addWidget(self.btn_market)
        ops_layout.addWidget(QLabel("Volume"))
        ops_layout.addWidget(self.volume_slider)
        ops_layout.addStretch()
        
        playlist_col.addLayout(header_row)
        playlist_col.addWidget(self.search_box)
        playlist_col.addWidget(self.playlist_widget)

        bottom_layout.addLayout(playlist_col, stretch=1)
        bottom_layout.addLayout(ops_layout)
        
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        
        # Connections
        self.btn_load.clicked.connect(self.load_files)
        self.btn_add_folder.clicked.connect(self.load_folder)
        self.btn_market.clicked.connect(self.open_marketplace)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_prev.clicked.connect(self.prev_track)
        self.btn_next.clicked.connect(self.next_track)
        self.btn_loop.clicked.connect(self.toggle_loop)
        self.btn_shuffle.clicked.connect(self.toggle_shuffle)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.btn_pin_playlist.toggled.connect(lambda v: self._update_playlist_size(animated=True))
        self.search_box.textChanged.connect(self._filter_playlist)
        self.playlist_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.playlist_widget.customContextMenuRequested.connect(self._on_playlist_context_menu)
        self.playlist_widget.installEventFilter(self)
        # Keyboard handled via keyPressEvent (Delete, Ctrl+F)
        
        # Track end checker
        self.end_timer = QTimer(self)
        self.end_timer.timeout.connect(self.check_end)
        self.end_timer.start(500)

        # Connect progress slider interactions
        self.progress_slider.sliderPressed.connect(lambda: setattr(self, '_seeking', True))
        self.progress_slider.sliderReleased.connect(self._on_progress_released)
        self.progress_slider.sliderMoved.connect(self._on_progress_moved)
        
        # Animation handle for playlist resizing
        self._playlist_anim = None

        # Ensure playlist box sizes responsively based on content and window
        self._update_playlist_size(animated=False)

        self._set_placeholder_art()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update playlist maximum height when window resizes
        self._update_playlist_size(animated=True)

    def _desired_playlist_height(self):
        # Calculate desired height based on number of items and available space
        max_visible = 8
        count = max(1, len(self.playlist))
        count = min(count, max_visible)

        # Try to get an accurate row height; fallback to 30
        row_h = self.playlist_widget.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 30

        padding = 16  # padding for top/bottom and internal margins
        desired = row_h * count + padding

        # Don't let playlist take more than ~35% of window height
        max_h = int(self.height() * 0.35)
        return min(desired, max_h)

    def _update_playlist_size(self, animated=True):
        target = self._desired_playlist_height()
        current = self.playlist_widget.maximumHeight()
        if not animated or current == 0:
            # immediate set
            self.playlist_widget.setMaximumHeight(target)
            return

        # Animate the maximumHeight property for a smooth resize
        try:
            if self._playlist_anim is not None and self._playlist_anim.state() == QPropertyAnimation.Running:
                self._playlist_anim.stop()

            anim = QPropertyAnimation(self.playlist_widget, b"maximumHeight")
            anim.setDuration(260)
            anim.setStartValue(current)
            anim.setEndValue(target)
            anim.start()
            # Keep a reference so it doesn't get GC'd
            self._playlist_anim = anim
        except Exception:
            # Fallback immediate
            self.playlist_widget.setMaximumHeight(target)

    # --------------------------
    # Reactive Background Logic
    # --------------------------
    def _animate_background(self):
        # Linearly interpolate current color towards target
        def lerp_color(c1, c2, t=0.05):
            r = c1.red() + (c2.red() - c1.red()) * t
            g = c1.green() + (c2.green() - c1.green()) * t
            b = c1.blue() + (c2.blue() - c1.blue()) * t
            return QColor(int(r), int(g), int(b))
        
        self.bg_color_1 = lerp_color(self.bg_color_1, self.target_color_1)
        self.bg_color_2 = lerp_color(self.bg_color_2, self.target_color_2)
        self.update() # triggers paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw the animated gradient background
        # Diagonal gradient for dynamic look
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, self.bg_color_1)
        grad.setColorAt(1.0, self.bg_color_2)
        
        painter.fillRect(self.rect(), grad)
        
        # Optional: Add a subtle radial "glow" in center
        rad = QRadialGradient(self.width()/2, self.height()/2, self.width()/2)
        rad.setColorAt(0, QColor(255, 255, 255, 10))
        rad.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), rad)

    # --------------------------
    # Player Logic
    # --------------------------
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Tracks", "", "Audio (*.mp3 *.wav *.ogg)")
        if files:
            for f in files:
                self.playlist.append(f)
                item = QListWidgetItem(basename(f))
                self.playlist_widget.addItem(item)
            if self.current_index == -1:
                self.current_index = 0
            # Adjust playlist height to fit new items (animated)
            self._update_playlist_size(animated=True)
            self._update_playlist_header()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            exts = {'.mp3', '.wav', '.ogg'}
            added = 0
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if Path(f).suffix.lower() in exts:
                        full = os.path.join(root, f)
                        self.playlist.append(full)
                        self.playlist_widget.addItem(QListWidgetItem(basename(full)))
                        added += 1
            if self.current_index == -1 and self.playlist:
                self.current_index = 0
            if added:
                self._update_playlist_size(animated=True)
                self._update_playlist_header()

    def open_marketplace(self):
        dlg = MarketplaceDialog(self, on_download=self._on_marketplace_download)
        dlg.exec()

    def _on_marketplace_download(self, filepath, add_to_playlist=True):
        # Called when marketplace finished download; optionally add to playlist
        if add_to_playlist and filepath:
            self.playlist.append(filepath)
            self.playlist_widget.addItem(QListWidgetItem(basename(filepath)))
            self._update_playlist_size(animated=True)
            self._update_playlist_header()

    def play_track(self, index):
        if 0 <= index < len(self.playlist):
            path = self.playlist[index]
            try:
                # If currently playing and crossfade is enabled, fade out
                xf = getattr(self, 'crossfade_slider', None)
                crossfade_ms = xf.value() if xf is not None else 0
                if pygame.mixer.music.get_busy() and crossfade_ms > 0:
                    try:
                        pygame.mixer.music.fadeout(crossfade_ms)
                    except Exception:
                        pass

                pygame.mixer.music.load(path)
                # Start with fade-in if crossfade_ms provided
                if crossfade_ms > 0:
                    pygame.mixer.music.play(fade_ms=crossfade_ms)
                else:
                    pygame.mixer.music.play()
                self.playing = True
                self.btn_play.setText("⏸ Pause")
                self.visualizer.active = True
                self.title_label.setText(basename(path))
                self.artist_label.setText("Now Playing")
                
                # Update Metadata & Reactive Colors
                self._update_metadata(path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _update_metadata(self, path):
        # Default colors
        c1, c2 = QColor(20, 0, 40), QColor(0, 20, 50)
        has_art = False
        
        if MUTAGEN_AVAILABLE:
            try:
                f = MutagenFile(path)
                if f and hasattr(f, 'tags'):
                    # Search for APIC (MP3) or Cover Art (FLAC/OGG logic could be added)
                    art_data = None
                    for tag in f.tags.values():
                        if getattr(tag, 'FrameID', '') == 'APIC':
                            art_data = tag.data
                            break
                    
                    if art_data:
                        pix = QPixmap()
                        pix.loadFromData(art_data)
                        
                        # Set Art
                        self.album_art.setPixmap(pix)
                        has_art = True
                        
                        # Extract Colors for Background
                        # Convert QPixmap -> PIL Image for analysis
                        img_data = pix.toImage()
                        # Helper to convert QImage to PIL
                        buffer = img_data.bits().asstring(img_data.sizeInBytes())
                        pil_img = Image.frombuffer("RGBA", (img_data.width(), img_data.height()), buffer, "raw", "BGRA", 0, 1)
                        
                        colors = get_dominant_colors(pil_img)
                        if len(colors) >= 2:
                            c1, c2 = colors[0], colors[1]
                        elif len(colors) == 1:
                            c1 = colors[0]
            except Exception as e:
                print("Metadata error:", e)

        if not has_art:
            self._set_placeholder_art()
            # Random Neon colors if no art
            c1 = QColor.fromHsv(random.randint(0, 360), 200, 100)
            c2 = QColor.fromHsv((c1.hue() + 40) % 360, 200, 50)
            
        # Set target colors for animation
        self.target_color_1 = c1
        self.target_color_2 = c2
        # Attempt to fetch track length (seconds)
        length = 0.0
        if MUTAGEN_AVAILABLE:
            try:
                f = MutagenFile(path)
                if f and hasattr(f, 'info') and hasattr(f.info, 'length'):
                    length = float(f.info.length)
            except Exception:
                length = 0.0

        if length == 0.0:
            try:
                # Fallback to pygame Sound length (may be heavy)
                snd = pygame.mixer.Sound(path)
                length = float(snd.get_length())
            except Exception:
                length = 0.0

        self.current_track_length = length
        if length > 0:
            self.progress_slider.setRange(0, int(math.ceil(length)))
            self.label_total.setText(self._format_time(length))
        else:
            # unknown length
            self.progress_slider.setRange(0, 100)
            self.label_total.setText("0:00")

    def _set_placeholder_art(self):
        pix = QPixmap(300, 300)
        pix.fill(QColor(20, 20, 20))
        painter = QPainter(pix)
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Segoe UI", 20))
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "🎵")
        painter.end()
        self.album_art.setPixmap(pix)

    def _format_time(self, seconds: float) -> str:
        try:
            s = int(seconds)
            m = s // 60
            ss = s % 60
            return f"{m}:{ss:02d}"
        except Exception:
            return "0:00"

    def _on_progress_moved(self, val):
        # While moving the slider, show provisional elapsed time
        if self.current_track_length > 0:
            # slider range is seconds when known
            secs = int(val)
            self.label_elapsed.setText(self._format_time(secs))
        else:
            # treat as percent
            pct = val / max(1, self.progress_slider.maximum())
            self.label_elapsed.setText(self._format_time(pct * self.current_track_length))

    def _on_progress_released(self):
        self._seeking = False
        val = self.progress_slider.value()
        # Compute seek seconds
        if self.current_track_length > 0:
            secs = float(val)
        else:
            # percentage into unknown length -> no-op
            secs = 0.0
        self._seek_to(secs)

    def _seek_to(self, seconds: float):
        # Try to seek using pygame's set_pos or play(start=)
        try:
            # stop and play from new position
            was_playing = self.playing
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.play(start=seconds)
            except TypeError:
                # older pygame may not accept named param
                pygame.mixer.music.play(0, seconds)
            self._current_play_offset = seconds
            if not was_playing:
                pygame.mixer.music.pause()
                self.playing = False
            else:
                self.playing = True
            # update UI
            self.label_elapsed.setText(self._format_time(seconds))
        except Exception:
            try:
                pygame.mixer.music.set_pos(seconds)
                self._current_play_offset = seconds
            except Exception:
                pass

    def _update_playlist_header(self):
        visible_count = sum(not self.playlist_widget.item(i).isHidden() for i in range(self.playlist_widget.count()))
        total = len(self.playlist)
        self.playlist_label.setText(f"Tracks ({visible_count}/{total})")
        # update size when content changes
        self._update_playlist_size(animated=True)

    def _filter_playlist(self, text):
        text = text.lower().strip()
        for i in range(self.playlist_widget.count()):
            item = self.playlist_widget.item(i)
            item_text = item.text().lower()
            item.setHidden(bool(text) and text not in item_text)
        self._update_playlist_header()

    def _on_playlist_context_menu(self, pos):
        menu = QMenu()
        remove_action = menu.addAction("Remove Selected")
        clear_action = menu.addAction("Clear Playlist")
        action = menu.exec(self.playlist_widget.mapToGlobal(pos))
        if action == remove_action:
            self._remove_selected()
        elif action == clear_action:
            self._clear_playlist()

    def _remove_selected(self):
        row = self.playlist_widget.currentRow()
        if row >= 0 and row < len(self.playlist):
            self.playlist.pop(row)
            self.playlist_widget.takeItem(row)
            if self.current_index >= len(self.playlist):
                self.current_index = len(self.playlist) - 1
            self._update_playlist_header()

    def _clear_playlist(self):
        self.playlist.clear()
        self.playlist_widget.clear()
        self.current_index = -1
        self.playing = False
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self._update_playlist_header()

    def eventFilter(self, source, event):
        # Hover expand/collapse for playlist for a more fluid feeling
        if source is self.playlist_widget:
            if event.type() == event.Type.Enter:
                # Expand unless pinned
                if not self.btn_pin_playlist.isChecked():
                    self._update_playlist_size(animated=True)
            elif event.type() == event.Type.Leave:
                if not self.btn_pin_playlist.isChecked():
                    # Collapse to minimal (1-2 rows) when not pinned
                    try:
                        # Temporarily set playlist to show 1 row height
                        current_max = self.playlist_widget.maximumHeight()
                        row_h = self.playlist_widget.sizeHintForRow(0) or 30
                        target = min(int(self.height() * 0.15), row_h * 2 + 12)
                        anim = QPropertyAnimation(self.playlist_widget, b"maximumHeight")
                        anim.setDuration(220)
                        anim.setStartValue(current_max)
                        anim.setEndValue(target)
                        anim.start()
                        self._playlist_anim = anim
                    except Exception:
                        pass
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        # Handle Delete to remove selected, Ctrl+F to focus search
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Delete:
            self._remove_selected()
            return
        # Ctrl+F
        if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_F:
            self.search_box.setFocus()
            return
        super().keyPressEvent(event)

    def toggle_play(self):
        if not self.playlist: return
        if self.playing:
            pygame.mixer.music.pause()
            self.playing = False
            self.visualizer.active = False
            self.btn_play.setText("▶ Play")
        else:
            if pygame.mixer.music.get_pos() > 0:
                pygame.mixer.music.unpause()
                self.playing = True
                self.visualizer.active = True
                self.btn_play.setText("⏸ Pause")
            else:
                self.play_track(self.current_index)

    def prev_track(self):
        if not self.playlist: return
        self.current_index = (self.current_index - 1) % len(self.playlist)
        self.play_track(self.current_index)

    def next_track(self):
        if not self.playlist: return
        if self.shuffle:
            self.current_index = random.randint(0, len(self.playlist)-1)
        else:
            self.current_index = (self.current_index + 1) % len(self.playlist)
        self.play_track(self.current_index)

    def toggle_loop(self):
        self.loop = not self.loop
        self.btn_loop.setStyleSheet(f"color: {'#00ff00' if self.loop else 'white'}")

    def toggle_shuffle(self):
        self.shuffle = not self.shuffle
        self.btn_shuffle.setStyleSheet(f"color: {'#00ff00' if self.shuffle else 'white'}")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def set_volume(self, val):
        pygame.mixer.music.set_volume(val / 100)

    def _on_item_double_click(self):
        self.current_index = self.playlist_widget.currentRow()
        self.play_track(self.current_index)

    def check_end(self):
        # Update progress display
        try:
            pos_ms = pygame.mixer.music.get_pos()
        except Exception:
            pos_ms = -1

        if pos_ms >= 0:
            elapsed = (pos_ms / 1000.0) + getattr(self, '_current_play_offset', 0.0)
            if not self._seeking:
                # update slider and label
                if self.current_track_length > 0:
                    # clamp
                    if elapsed > self.current_track_length:
                        elapsed = self.current_track_length
                    self.progress_slider.setValue(int(elapsed))
                else:
                    # unknown length - update percent-like
                    maxv = self.progress_slider.maximum()
                    self.progress_slider.setValue(int((elapsed % 1) * maxv))
                self.label_elapsed.setText(self._format_time(elapsed))

        if self.playing and not pygame.mixer.music.get_busy():
            # Track ended
            if self.loop:
                self.play_track(self.current_index)
            else:
                self.next_track()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeonPlayer()
    window.show()
    sys.exit(app.exec())