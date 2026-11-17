using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace DataVoleyLauncher
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string scriptPath = Path.Combine(baseDir, "run_app.py");

            if (!File.Exists(scriptPath))
            {
                MessageBox.Show(
                    "No se encontro run_app.py en:\n" + scriptPath,
                    "DataVoley Studio — Error de Lanzador",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            string pythonExe = FindPythonExecutable();
            if (string.IsNullOrEmpty(pythonExe))
            {
                MessageBox.Show(
                    "No se detecto una instalacion de Python 3 en el sistema.\nPor favor instala Python 3.11+ y asegurate de anadirlo al PATH.",
                    "DataVoley Studio — Python Requerido",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning
                );
                return;
            }

            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "\"" + scriptPath + "\"",
                    WorkingDirectory = baseDir,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error al iniciar DataVoley Studio:\n" + ex.Message,
                    "DataVoley Studio",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }

        private static string FindPythonExecutable()
        {
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string[] candidates = new string[]
            {
                Path.Combine(localAppData, @"Programs\Python\Python312\pythonw.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python313\pythonw.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python311\pythonw.exe"),
                "pythonw.exe",
                "python.exe"
            };

            foreach (string candidate in candidates)
            {
                if (Path.IsPathRooted(candidate))
                {
                    if (File.Exists(candidate))
                        return candidate;
                }
                else
                {
                    string pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
                    foreach (string dir in pathEnv.Split(Path.PathSeparator))
                    {
                        try
                        {
                            string full = Path.Combine(dir.Trim(), candidate);
                            if (File.Exists(full))
                                return full;
                        }
                        catch { }
                    }
                }
            }
            return null;
        }
    }
}
