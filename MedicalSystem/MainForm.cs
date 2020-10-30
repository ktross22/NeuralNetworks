﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetworks;

namespace MedicalSystem
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        private void MainForm_Load(object sender, EventArgs e)
        {

        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var aboutForm = new AboutForm();
            aboutForm.ShowDialog();
        }

        private void imageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if( openFileDialog.ShowDialog() == DialogResult.OK)
            {
                var pictureConvecter = new PictureConverter();
                var inputs= pictureConvecter.Convert(openFileDialog.FileName);
                var result= Program.Controller.ImageNetwork.Predict(inputs).Output;
                

            }
        }

        private void enterToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var enterdataForm = new EnterData();
            var result = enterdataForm.ShowForm();
        }
    }
}
