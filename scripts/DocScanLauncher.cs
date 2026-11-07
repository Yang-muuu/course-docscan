using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("DocScan 文档扫描")]
[assembly: AssemblyVersion("0.2.0.0")]

internal static class DocScanLauncher
{
    [STAThread]
    private static int Main()
    {
        try
        {
            string directory = AppDomain.CurrentDomain.BaseDirectory;
            string interpreter = Path.Combine(directory, "runtime", "pythonw.exe");
            if (!File.Exists(interpreter))
                throw new FileNotFoundException("请完整解压压缩包，再从解压目录打开 DocScan.exe。", interpreter);

            var start = new ProcessStartInfo(interpreter, "-X utf8 -B -m docscan.desktop")
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = directory
            };
            using (Process app = Process.Start(start))
            {
                app.WaitForExit();
                if (app.ExitCode != 0)
                    MessageBox.Show("启动未完成。请重新完整解压程序包。错误代码 " + app.ExitCode,
                        "文档扫描", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return app.ExitCode;
            }
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "文档扫描 · 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
