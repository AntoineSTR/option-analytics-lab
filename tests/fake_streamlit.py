"""Minimal stand-in for the streamlit module so every page script can be executed
headless (smoke test of the UI code paths without a browser).

Usage: python tests/smoke_ui.py
"""
import contextlib
import sys
import types


class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(_st, name)


class _ColCfg:
    def __getattr__(self, name):
        return lambda *a, **k: None


class Stop(Exception):
    pass


class _ST(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _State()
        self.column_config = _ColCfg()
        self.press = set()          # button keys / labels to press
        self.files = {}             # uploader key -> file-like
        self.calls = []
        self.radio_choice = {}

    # ---- layout
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def tabs(self, labels):
        return [_Ctx() for _ in labels]

    def container(self, **k):
        return _Ctx()

    def expander(self, *a, **k):
        return _Ctx()

    def spinner(self, *a, **k):
        return _Ctx()

    sidebar = property(lambda self: _Ctx())

    # ---- input widgets
    def _val(self, key, default):
        if key is not None:
            if key not in self.session_state:
                self.session_state[key] = default
            return self.session_state[key]
        return default

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, **k):
        if value is None:
            value = min_value if min_value is not None else 0.0
        return self._val(key, value)

    def text_input(self, label, value="", key=None, **k):
        return self._val(key, value)

    def selectbox(self, label, options, index=0, key=None, format_func=None, **k):
        options = list(options)
        return self._val(key, options[index] if options else None)

    def radio(self, label, options, index=0, key=None, **k):
        options = list(options)
        if key in self.radio_choice:
            return self.radio_choice[key]
        return self._val(key, options[index])

    def checkbox(self, label, value=False, key=None, **k):
        if key in self.press:
            return True
        return self._val(key, value)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        return self._val(key, value if value is not None else min_value)

    def select_slider(self, label, options=(), value=None, key=None, **k):
        return self._val(key, value if value is not None else list(options)[0])

    def multiselect(self, label, options, default=None, key=None, **k):
        return self._val(key, default or [])

    def button(self, label, key=None, on_click=None, **k):
        hit = key in self.press or label in self.press
        if hit and on_click:
            on_click()
        return hit

    def download_button(self, label, data, *a, **k):
        assert data is not None and len(data) > 0, f"empty download {label}"
        self.calls.append(("download", label, len(data)))
        return False

    def file_uploader(self, label, key=None, **k):
        return self.files.get(key)

    def data_editor(self, df, key=None, **k):
        return df.copy()

    # ---- output
    def _noop(self, *a, **k):
        self.calls.append(("out",) + tuple(str(x)[:60] for x in a[:1]))

    title = header = subheader = markdown = caption = write = info = success = warning = error = latex = divider = \
        page_link = code = json = text = _noop

    def metric(self, label, value, delta=None, **k):
        self.calls.append(("metric", label, value))

    def dataframe(self, data, **k):
        # force a Styler to render so format strings are validated
        if hasattr(data, "to_html"):
            data.to_html()
        self.calls.append(("df",))

    def plotly_chart(self, fig, **k):
        fig.to_json()
        self.calls.append(("chart",))

    def stop(self):
        raise Stop()

    def rerun(self):
        raise Stop()

    # ---- decorators / misc
    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    def dialog(self, title, **k):
        return lambda f: f

    def set_page_config(self, **k):
        pass

    def Page(self, path, **k):
        return path

    def navigation(self, pages):
        return types.SimpleNamespace(run=lambda: None)


_st = _ST()


def install():
    sys.modules["streamlit"] = _st
    return _st
