
# coding: utf-8

# In[1]:

import openturns as ot
from openturns.viewer import View
import numpy as np
import pylab as pl
get_ipython().magic(u'pylab --no-import-all inline')


# # Step A: define the physical model

# In[2]:

model = ot.NumericalMathFunction(["F", "E", "L", "I"], ["v"], ["F*L^3/(3*E*I)"])


# In[3]:

m = np.array([30000,3.38744e+07,255,397.5])
model(m)


# # Step B: specify the input random vector # 

# In[4]:

sample_E = ot.NumericalSample.ImportFromCSVFile("sample_E.csv") 
kernel_smoothing = ot.KernelSmoothing(ot.Normal())
bandwidth = kernel_smoothing.computeSilvermanBandwidth(sample_E)
E = kernel_smoothing.build(sample_E, bandwidth)
E.setDescription(['Young modulus'])


# In[5]:

F = ot.LogNormal(30000, 9000, 15000, ot.LogNormal.MUSIGMA)
F.setDescription(['Load'])


# In[6]:

L = ot.Uniform(250, 260)
L.setDescription(['Length'])


# In[7]:

I = ot.Beta(2.5, 4, 310, 450)
I.setDescription(['Inertia'])


# In[8]:

marginal_distributions = [F, E, L, I]


# In[9]:

SR_cor = ot.CorrelationMatrix(len(marginal_distributions))
SR_cor[2, 3] = -0.2
copula = ot.NormalCopula(ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(SR_cor))


# In[10]:

input_distribution = ot.ComposedDistribution(marginal_distributions, copula)


# In[11]:

input_random_vector = ot.RandomVector(input_distribution)


# # Step C: uncertainty propagation
# 
# A polynomial chaos (PC) metamodel is used.
# 
# ## Specification of the PC basis
# 
# We use "dedicated" families of polynomials, except for the lognormal PDF which is not defined uniquely by its moment sequence.

# In[12]:

univariate_polynomials_bases = [
    ot.HermiteFactory(),
    ot.StandardDistributionPolynomialFactory(E),
    ot.StandardDistributionPolynomialFactory(L),
    ot.StandardDistributionPolynomialFactory(I)]


# In[13]:

enumerate_function = ot.EnumerateFunction(input_distribution.getDimension())


# In[14]:

multivariate_polynomial_basis = ot.OrthogonalProductPolynomialFactory(univariate_polynomials_bases, enumerate_function)


# ## Construction of the PC approximation

# In[15]:

basis_sequence_factory = ot.LAR()


# In[16]:

fitting_algorithm = ot.CorrectedLeaveOneOut()


# In[17]:

approximation_algorithm = ot.LeastSquaresMetaModelSelectionFactory()


# In[18]:

design_of_experiments = ot.MonteCarloExperiment(1000)


# In[19]:

evaluation_strategy = ot.LeastSquaresStrategy(design_of_experiments, approximation_algorithm)


# In[20]:

total_degree = 3
n_terms =  enumerate_function.getStrataCumulatedCardinal(total_degree)


# In[21]:

basis_truncature_strategy = ot.FixedStrategy(multivariate_polynomial_basis, n_terms)


# In[22]:

PCE_algorithm = ot.FunctionalChaosAlgorithm(model, input_distribution, basis_truncature_strategy, evaluation_strategy)
PCE_algorithm.run()


# In[23]:

PCE_result = PCE_algorithm.getResult()
metamodel = PCE_result.getMetaModel()


# In[24]:

metamodel(input_distribution.getSample(1))


# ## Assess the metamodel goodness-of-fit

# Question: What happens with the following metrics if we replace the adapted polynomials with Legendre or Hermite polynomials?

# ### Leave-one-out estimate of the $R^2$ score

# In[25]:

R2 = 1. - PCE_result.getRelativeErrors()[0]
print("\nR2 coefficient: %.6f \n" % R2)


# ### Adequation plot

# In[26]:

input_sample = input_random_vector.getSample(2000)


# In[27]:

model_sample = model(input_sample)
model_sample.setName("Actual output")


# In[28]:

metamodel_sample = metamodel(input_sample)
metamodel_sample.setName("Chaos-based output")


                linear_regression_model = ot.LinearModelFactory().build(model_sample, metamodel_sample)
_ = View(ot.VisualTest.DrawLinearModel(model_sample, metamodel_sample, linear_regression_model))
                
# ### Superposition of the model and the metamodel in 1D cuts

# In[29]:

input_number = 0
input_margin = input_distribution.getMarginal(input_number)
model_cut = model.draw(input_number,
                       0,
                       input_distribution.getMean(),
                       input_margin.computeQuantile(.025)[0],
                       input_margin.computeQuantile(.975)[0],
                       100).getDrawable(0)
metamodel_cut = metamodel.draw(input_number,
                               0,
                               input_distribution.getMean(),
                               input_margin.computeQuantile(.025)[0],
                               input_margin.computeQuantile(.975)[0],
                               100).getDrawable(0)
graph = ot.Graph()
graph.add([model_cut, metamodel_cut])
graph.setColors(["blue", "red"])
graph.setLegends(["Model", "Metamodel"])
graph.setXTitle(input_margin.getDescription()[0])
graph.setYTitle(model.getOutputDescription()[0])
_ = View(graph)


# Computation of the output mean and standard deviation from the chaos coefficients

# ## Use chaos for surrogate-based statistical analyses
# 
# First let's make a random vector out of the polynomial chaos metamodel.

# In[30]:

chaos_based_output_random_vector = ot.FunctionalChaosRandomVector(PCE_result)


# ### Compute moments

# In[31]:

print("Mean: %.2f" % chaos_based_output_random_vector.getMean()[0])


# ### Computation of the Sobol indices from the chaos coefficients

# In[32]:

for i in range(input_distribution.getDimension()):
    print "First-order Sobol index for %s is %.2f" % (
        input_distribution.getDescription()[i], chaos_based_output_random_vector.getSobolIndex(i))


# In[33]:

for i in range(input_distribution.getDimension()):
    print "Total-order Sobol index for %s is %.2f" % (
        input_distribution.getDescription()[i], chaos_based_output_random_vector.getSobolTotalIndex(i))


# ### Reliability analysis using a (large) Monte Carlo sample of the chaos approximation

# In[34]:

event = ot.Event(chaos_based_output_random_vector, ot.Greater(), 30.) 
event.setName("Deviation > %s cm" % event.getThreshold())


# ### PDF and CDF approximations

# In[35]:

sample = chaos_based_output_random_vector.getSample(1000)
fig = pl.figure(figsize=(12, 4))
ax_pdf = fig.add_subplot(1, 2, 1)
hist_E = ot.VisualTest_DrawHistogram(sample)
hist_E.setColors(["gray"])
hist_E.setLegends(["Histogram"])
_ = View(hist_E, figure=fig, axes=[ax_pdf])
ax_cdf = fig.add_subplot(1, 2, 2)
ecdf_E = ot.VisualTest_DrawEmpiricalCDF(sample, sample.getMin()[0], sample.getMax()[0])
ecdf_E.setLegends(["Empirical CDF"])
_ = View(ecdf_E, figure=fig, axes=[ax_cdf])


# In[36]:

MCS_algorithm = ot.MonteCarlo(event) 
MCS_algorithm.setMaximumOuterSampling(40000) 
MCS_algorithm.setBlockSize(1) 
MCS_algorithm.setMaximumCoefficientOfVariation(.1)
MCS_algorithm.run()
MCS_result = MCS_algorithm.getResult()


# In[37]:

print "Proability estimate = %.2e" % MCS_result.getProbabilityEstimate() 
print "Coefficient of variation = %.2f" % MCS_result.getCoefficientOfVariation()
print "95%% Confidence Interval = [%.2e, %.2e]" % (MCS_result.getProbabilityEstimate() - .5 * MCS_result.getConfidenceLength(0.95),
                                                   MCS_result.getProbabilityEstimate() + .5 * MCS_result.getConfidenceLength(0.95))


# In[38]:

_ = View(MCS_algorithm.drawProbabilityConvergence(.95))


# In[38]:



