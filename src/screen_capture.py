"""画面キャプチャ＋入力注入のバックエンド抽象 (M6 / 実機 Wayland 対応).

realtime ループは「指定 bbox の BGR フレームをもらう」「指定座標をクリックする」だけを
必要とする。環境に応じて 2 実装を切り替える:

  - MssCapture:    X11/Xorg。mss でルートから grab。クリックは pynput(XTEST)。
  - PortalCapture: GNOME など Wayland。xdg-desktop-portal の RemoteDesktop +
                   ScreenCast を 1 セッションで開き、画面は PipeWire ノードを
                   GStreamer `pipewiresrc` で受け、クリックはコンポジタ経由で
                   NotifyPointerMotionAbsolute/Button により注入する。
                   ※ Wayland では mss は全黒、pynput(XTEST) のクリックは
                     ウィンドウに届かないため、両方ともこの実装が必須。

共通 IF:
    cap.geometry() -> (left, top, width, height)
    cap.grab(bbox) -> np.ndarray (H, W, 3) BGR   # bbox=(x,y,w,h) 絶対座標。None で全体
    cap.can_inject -> bool                        # True なら click を内蔵注入で行える
    cap.click(xy)                                 # can_inject のときのみ
    cap.close()

`open_capture(backend="auto")` が環境を見て適切な実装を返す。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

BTN_LEFT = 0x110  # evdev BTN_LEFT


def _is_wayland() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY")) or \
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


# --- mss (X11) -----------------------------------------------------------

class MssCapture:
    can_inject = False

    def __init__(self):
        import mss
        self._sct = mss.MSS()
        m = self._sct.monitors[1]
        self._geom = (m["left"], m["top"], m["width"], m["height"])

    def geometry(self):
        return self._geom

    def grab(self, bbox=None):
        if bbox is None:
            bbox = self._geom
        x, y, w, h = bbox
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3]  # BGRA -> BGR

    def click(self, xy):  # pragma: no cover - never used (can_inject False)
        raise NotImplementedError

    def close(self):
        try:
            self._sct.close()
        except Exception:  # noqa: BLE001
            pass


# --- Portal (Wayland: ScreenCast 取得 + RemoteDesktop 注入) ---------------

class PortalCapture:
    """xdg-desktop-portal RemoteDesktop + ScreenCast を 1 セッションで使う。

    生成時に GNOME のダイアログ（画面共有＋リモート操作の許可）が出る。許可後は
    画面取得とポインタ注入の両方が同じセッション/座標系で行える。
    """

    BUS = "org.freedesktop.portal.Desktop"
    OBJ = "/org/freedesktop/portal/desktop"
    RD = "org.freedesktop.portal.RemoteDesktop"
    SC = "org.freedesktop.portal.ScreenCast"
    REQ = "org.freedesktop.portal.Request"

    can_inject = True

    def __init__(self, cursor: bool = True, timeout_s: float = 120.0):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib, Gio
        self._Gst, self._GLib, self._Gio = Gst, GLib, Gio

        self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self._sender = self._bus.get_unique_name()[1:].replace(".", "_")
        self._loop = GLib.MainLoop()
        self._st = {"n": 0, "session": None, "node": None, "fd": None,
                    "geom": None, "error": None}

        self._negotiate(cursor)
        if self._st["error"]:
            raise RuntimeError(f"portal 失敗: {self._st['error']}")

        Gst.init(None)
        desc = (f"pipewiresrc fd={self._st['fd']} path={self._st['node']} ! "
                f"videoconvert ! video/x-raw,format=BGRx ! "
                f"appsink name=sink max-buffers=1 drop=true sync=false")
        self._pipeline = Gst.parse_launch(desc)
        self._sink = self._pipeline.get_by_name("sink")
        self._pipeline.set_state(Gst.State.PLAYING)
        self._timeout = int(timeout_s)
        self._warm()

    # -- portal D-Bus helpers --
    def _token(self, p):
        self._st["n"] += 1
        return f"{p}{self._st['n']}"

    def _req_path(self, tok):
        return f"/org/freedesktop/portal/desktop/request/{self._sender}/{tok}"

    def _call(self, iface, method, body, on_response):
        """Request パターンのメソッド（CreateSession/SelectX/Start）。"""
        GLib, Gio = self._GLib, self._Gio
        htok = self._token("req")
        path = self._req_path(htok)
        sub = {"id": None}

        def on_signal(conn, snd, obj_path, ifc, sig, params):
            self._bus.signal_unsubscribe(sub["id"])
            code, results = params.unpack()
            if code != 0:
                self._st["error"] = f"{method} code={code}（拒否/タイムアウト?）"
                self._loop.quit()
                return
            on_response(results)

        sub["id"] = self._bus.signal_subscribe(
            self.BUS, self.REQ, "Response", path, None,
            Gio.DBusSignalFlags.NONE, on_signal)
        self._bus.call(self.BUS, self.OBJ, iface, method, body(htok),
                       GLib.VariantType.new("(o)"), Gio.DBusCallFlags.NONE,
                       -1, None, None)

    def _notify(self, method, body_variant):
        """Notify* 系（戻り値なし・Request を介さない即時メソッド）。"""
        self._bus.call(self.BUS, self.OBJ, self.RD, method, body_variant,
                       None, self._Gio.DBusCallFlags.NONE, -1, None, None)

    def _negotiate(self, cursor):
        GLib = self._GLib

        def create():
            def body(htok):
                opts = {"handle_token": GLib.Variant("s", htok),
                        "session_handle_token": GLib.Variant("s", self._token("sess"))}
                return GLib.Variant("(a{sv})", (opts,))
            self._call(self.RD, "CreateSession", body, on_session)

        def on_session(results):
            self._st["session"] = results["session_handle"]
            select_devices()

        def select_devices():
            def body(htok):
                opts = {"handle_token": GLib.Variant("s", htok),
                        "types": GLib.Variant("u", 2)}  # POINTER
                return GLib.Variant("(oa{sv})", (self._st["session"], opts))
            self._call(self.RD, "SelectDevices", body, lambda r: select_sources())

        def select_sources():
            def body(htok):
                opts = {"handle_token": GLib.Variant("s", htok),
                        "types": GLib.Variant("u", 1),       # MONITOR
                        "multiple": GLib.Variant("b", False),
                        "cursor_mode": GLib.Variant("u", 2 if cursor else 1)}
                return GLib.Variant("(oa{sv})", (self._st["session"], opts))
            self._call(self.SC, "SelectSources", body, lambda r: start())

        def start():
            def body(htok):
                opts = {"handle_token": GLib.Variant("s", htok)}
                return GLib.Variant("(osa{sv})", (self._st["session"], "", opts))
            self._call(self.RD, "Start", body, on_started)

        def on_started(results):
            node_id, props = results["streams"][0]
            self._st["node"] = node_id
            pos = props.get("position", (0, 0))
            size = props.get("size", (0, 0))
            self._st["geom"] = (int(pos[0]), int(pos[1]), int(size[0]), int(size[1]))
            open_remote()

        def open_remote():
            Gio = self._Gio
            opts = GLib.Variant("(oa{sv})", (self._st["session"], {}))
            ret, fdlist = self._bus.call_with_unix_fd_list_sync(
                self.BUS, self.OBJ, self.SC, "OpenPipeWireRemote", opts,
                GLib.VariantType.new("(h)"), Gio.DBusCallFlags.NONE, -1, None, None)
            self._st["fd"] = fdlist.get(ret.unpack()[0])
            self._loop.quit()

        print(">>> GNOME のダイアログで画面共有＋操作を許可してください...", file=sys.stderr)
        create()
        self._loop.run()

    # -- frame --
    def _warm(self):
        Gst = self._Gst
        sample = self._sink.emit("try-pull-sample", self._timeout * Gst.SECOND)
        if not sample:
            raise RuntimeError("最初のフレームを取得できませんでした")
        st = sample.get_caps().get_structure(0)
        self._w = st.get_value("width")
        self._h = st.get_value("height")
        gx, gy, _, _ = self._st["geom"]
        self._st["geom"] = (gx, gy, self._w, self._h)

    def geometry(self):
        return self._st["geom"]

    def grab(self, bbox=None):
        Gst = self._Gst
        sample = self._sink.emit("try-pull-sample", self._timeout * Gst.SECOND)
        if not sample:
            raise RuntimeError("フレーム取得タイムアウト")
        buf = sample.get_buffer()
        ok, mi = buf.map(Gst.MapFlags.READ)
        try:
            frame = np.frombuffer(mi.data, np.uint8).reshape(self._h, self._w, 4)[:, :, :3]
            frame = np.ascontiguousarray(frame)
        finally:
            buf.unmap(mi)
        if bbox is None:
            return frame
        gx, gy, _, _ = self._st["geom"]
        x, y, w, h = bbox
        rx, ry = x - gx, y - gy
        return np.ascontiguousarray(frame[ry:ry + h, rx:rx + w])

    # -- input injection --
    def click(self, xy):
        GLib = self._GLib
        gx, gy, _, _ = self._st["geom"]
        x, y = float(xy[0] - gx), float(xy[1] - gy)  # stream 座標系（モニタ相対）
        node = self._st["node"]
        empty = GLib.Variant("a{sv}", {})
        self._notify("NotifyPointerMotionAbsolute",
                     GLib.Variant("(oa{sv}udd)", (self._st["session"], {}, node, x, y)))
        time.sleep(0.03)
        self._notify("NotifyPointerButton",
                     GLib.Variant("(oa{sv}iu)", (self._st["session"], {}, BTN_LEFT, 1)))
        time.sleep(0.03)
        self._notify("NotifyPointerButton",
                     GLib.Variant("(oa{sv}iu)", (self._st["session"], {}, BTN_LEFT, 0)))

    def close(self):
        try:
            self._pipeline.set_state(self._Gst.State.NULL)
        except Exception:  # noqa: BLE001
            pass


def open_capture(backend: str = "auto", cursor: bool = True):
    """backend: 'auto' | 'mss' | 'portal'。auto は Wayland なら portal。"""
    if backend == "auto":
        backend = "portal" if _is_wayland() else "mss"
    if backend in ("portal", "pipewire"):
        return PortalCapture(cursor=cursor)
    if backend == "mss":
        return MssCapture()
    raise ValueError(f"未知の capture backend: {backend}")
