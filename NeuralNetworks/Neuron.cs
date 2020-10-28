
using System;
using System.Collections.Generic;
using System.Xml;

namespace NeuralNetworks
{
    public class Neuron
    {
        public List<double> Weigths { get; }
        public  List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }


        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weigths = new List<double>();
            Inputs = new List<double>();

            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weigths.Add(1);
                }
                else
                {
                    Weigths.Add(rnd.NextDouble());
                    
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            for(int i=0; i< Inputs.Count;i++)
            {
                Inputs[i] = inputs[i];
            }

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

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }


        public void Learn(double error, double learningRate)
         {
            if(NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for(int i =0; i< Weigths.Count; i++)
            {
                var weight = Weigths[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weigths[i] = newWeight;


            }


        }

        public override string ToString()
        {
            return Output.ToString();
        }

    }
}
