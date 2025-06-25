import os

from typing import Any, List

from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from gcsfs import GCSFileSystem


class FileSystemIO:
    def __init__(
        self,
        fs: AbstractFileSystem,
        root_path: str,
    ) -> None:
        self._fs = fs
        self._root_path = root_path

    def get_video(self, path: str) -> str | bytes:
        source = os.path.join(self._root_path, path)
        with self._fs.open(source, "rb") as input_file:
            file_content = input_file.read()

        return file_content

    def save_video(self, video: bytes, path: str) -> None:
        destination = os.path.join(self._root_path, path)
        with self._fs.open(destination, "wb") as file:
            file.write(video)

    def remove_video(self, path: str) -> None:
        source = os.path.join(self._root_path, path)
        if self._fs.exists(source):
            self._fs.rm(source)

    def list_files(self, path: str = "", rec: bool = False) -> List[str]:
        dir_path = os.path.join(self._root_path, path)
        if not self._fs.exists(dir_path):
            return []

        if rec:
            files = self._fs.find(dir_path, withdirs=False)
        else:
            files = self._fs.ls(dir_path)
            files = [f for f in files if f != dir_path and not f.endswith("/")]

        files = ["/".join(f.split("/")[1:]) for f in files]
        return files


class LocalIO(FileSystemIO):
    """Version that writes directly to local."""

    def __init__(self, root_path: str) -> None:
        fs = LocalFileSystem(auto_mkdir=True)
        super().__init__(fs, root_path)


class GcsIO(FileSystemIO):
    """Specialized version of file system for reading and loading data from GCS"""

    def __init__(
        self,
        project_id: str,
        root_path: str,
        credentials: Any | None = None,
        **kwargs
    ) -> None:
        """Constructor
        Args:
            project_id: Id of the project.
            root_path: Root path for storing, loading the data.
            credentials: Either a Credentials object or a json path to a credentials file. By
                default is None and takes the credentials from google.auth.defailt.
            kwargs: Kwargs for the GCSFileSystem.
        """
        from google.auth import default

        credentials = credentials or default()[0]

        fs = GCSFileSystem(project=project_id, token=credentials)
        super().__init__(fs, root_path, **kwargs)
