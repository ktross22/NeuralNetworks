
using System;
using System.Collections.Generic;
using System.Xml;

namespace NeuralNetworks
{
    public class Neuron
    {
        public List<double> Weigths { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weigths = new List<double>();

            for(int i = 0; i< inputCount; i++)
            {
                Weigths.Add(1);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            var sum = 0.0;
            for(int i =0; i< inputs.Count; i++)
            {
                sum += inputs[i] * Weigths[i];
            }
            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }
        public void SetWigths(params double[] weigths)
        {
            //TODO: удалить после добавления возможности обучения сети
            for(int i=0; i< weigths.Length;i++)
            {
                Weigths[i] = weigths[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }

    }
}
