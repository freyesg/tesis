public void run() {
    while (currentGeneration <= LEEAParams.MAXGENERATIONS) {
        currentGeneration++;
        if (currentGeneration % SharedParams.SharedParams.TRACKFITNESSSTRIDE == 0)
            SharedParams.SharedParams.PERFORMFULLEVAL = true;
        else
            SharedParams.SharedParams.PERFORMFULLEVAL = false;
            performOneGeneration();
    }
}

private void performOneGeneration() {
	// produce offspring
	produceOffspring();

	// speciate the new kids
	speciatePopulation();

	// evaluate the new population
	evaluatePopulation();

	// update evolution parameters
	MUTATIONPOWER *= decayRate;
	MUTATIONRATE *= rateDecayRate;
}

private void produceOffspring()
{
	double[] offspringCount = new double[LEEAParams.SPECIESCOUNT];

	// generate species-specific genome lists
	List<QEAGenome>[] speciesGenomes = new List<QEAGenome>[LEEAParams.SPECIESCOUNT];
	for (int i = 0; i < LEEAParams.SPECIESCOUNT; i++)
		speciesGenomes[i] = new List<QEAGenome>();

	for (int i = 0; i < genomeList.Count; i++)
		speciesGenomes[genomeList[i].Species].Add(genomeList[i]);

	// determine offspring count for each species
	if (LEEAParams.SPECIESCOUNT == 1)
	{
		offspringCount[0] = LEEAParams.POPSIZE;
	}
	else
	{
		double[] specieFitness = new double[LEEAParams.SPECIESCOUNT];
		// calculate species stats to determine how many offspring each species is granted

		for (int i = 0; i < LEEAParams.SPECIESCOUNT; i++)
		{
			for (int j = 0; j < speciesGenomes[i].Count; j++)
			{
				specieFitness[i] += speciesGenomes[i][j].Fitness;
			}

			if (speciesGenomes[i].Count != 0)
				specieFitness[i] /= speciesGenomes[i].Count;
		}


		for (int i = 0; i < LEEAParams.SPECIESCOUNT; i++)
			offspringCount[i] = Math.Round(LEEAParams.POPSIZE * specieFitness[i] / specieFitness.Sum());

		// rounding error could leave us a few short or long of the pop size, trim or fill to reach population size
		RouletteWheelLayout rwl = new RouletteWheelLayout(offspringCount);
		while (offspringCount.Sum() > LEEAParams.POPSIZE)
		{
			int index = RouletteWheel.SingleThrow(rwl, r);
			if (offspringCount[index] > 1)
				offspringCount[index]--;
		}

		while (offspringCount.Sum() < LEEAParams.POPSIZE)
		{
			int index = RouletteWheel.SingleThrow(rwl, r);
			offspringCount[index]++;
		}
	}

	List<QEAGenome> newGeneration = new List<QEAGenome>();

	// generate offspring for each species
	// parallelism doesn't work if speciescount = 1 here!

	for (int i = 0; i < LEEAParams.SPECIESCOUNT; i++)
	{
		if (offspringCount[i] > 0)
		{
			// sort the genome list by fitness
			Comparison<QEAGenome> comparison = (x, y) => y.Fitness.CompareTo(x.Fitness);
			speciesGenomes[i].Sort(comparison);

			// determine the top X individuals that we will select from
			int selectionNumber = (int)(speciesGenomes[i].Count * LEEAParams.SELECTIONPROPORTION);
			if (selectionNumber == 0)
				selectionNumber = 1;

			// build list of probabilities based on fitness
			double[] probabilities = new double[selectionNumber];

			for (int j = 0; j < probabilities.Length; j++)
				probabilities[j] = speciesGenomes[i][j].Fitness;

			RouletteWheelLayout rw = new RouletteWheelLayout(probabilities);

			// build a list of matings to be performed.  This must be done outside of the parallelized section.
			int[][] matings = new int[(int)offspringCount[i]][];
			for (int j = 0; j < matings.Length; j++)
			{
				matings[j] = new int[2];

				// select main parent
				int index = RouletteWheel.SingleThrow(rw, r);

				if (r.NextDouble() < LEEAParams.SEXPROPORTION && probabilities.Length > 1) // can't have sexual reproduction if this species only has a single member
				{
					matings[j][0] = index;

					int parent2 = index;
					while(parent2 == index)
						parent2 = RouletteWheel.SingleThrow(rw, r);
					matings[j][1] = parent2;
				}
				else
				{
					matings[j][0] = index;
					matings[j][1] = int.MinValue;
				}
			}

			Parallel.For(0, matings.Length, po, j =>
			//for (int j = 0; j < matings.Length; j++)
			{
				// mutate
				QEAGenome child;
				if (matings[j][1] > int.MinValue)
				{
					// sexual reproduction
					child = speciesGenomes[i][matings[j][0]].createOffspring(speciesGenomes[i][matings[j][1]]);
					child.Fitness = (speciesGenomes[i][matings[j][0]].Fitness + speciesGenomes[i][matings[j][1]].Fitness) / 2;
				}
				else
				{
					child = speciesGenomes[i][matings[j][0]].createOffspring();
					child.Fitness = speciesGenomes[i][matings[j][0]].Fitness;
				}

				lock (newGeneration)
				{
					newGeneration.Add(child);
				}
			});
		}
	}

	// encourage the garbage collector to free up some memory
	foreach (QEAGenome g in genomeList) {
		g.weights = null;
	}
	genomeList = null;


	genomeList = newGeneration;
}


private void evaluatePopulation()
{
    evaluator.NewGeneration();

    for (int i = 0; i < workers.Length; i++)
    {
        workers[i].setGenomeList(genomeList);
        threads[i] = new Thread(new ThreadStart(workers[i].DoWork));
        threads[i].Start();
    }

    EvaluationWorker mainWorker = new EvaluationWorker(start, genomeList.Count, genomeList, evaluator);
    mainWorker.DoWork();

    for (int i = 0; i < threads.Length; i++)
        threads[i].Join();

    updateStats();
}

private void speciatePopulation() {
    if (LEEAParams.SPECIESCOUNT == 1) return;

    double[][] centroids = new double[LEEAParams.SPECIESCOUNT][];
    double[][] flatWeights = new double[genomeList.Count][];

    int connectionCount = genomeList[0].weights.SelectMany(s => s).SelectMany(s => s).ToArray().Length;

    // flatten the weight arrays to them more efficient to work with
	//for (int i = 0; i < genomeList.Count; i++)
    Parallel.For(0, genomeList.Count, i =>{
        flatWeights[i] = new double[connectionCount];
            Array.Copy(genomeList[i].weights.SelectMany(s => s).SelectMany(s => s).ToArray(), flatWeights[i], connectionCount);
    });

    // set the centroid to x random individuals in the population.  This ensures each cluster has at least one individual.
    for (int i = 0; i < LEEAParams.SPECIESCOUNT; i++) {
        centroids[i] = new double[connectionCount];
        Array.Copy(flatWeights[i], centroids[i], connectionCount);
    }

    int kMeansCount = 0;
    bool changed = true;

        // so long as individuals are being reassigned, continue the kmeans loop.. max iterations of 5
    while (changed && kMeansCount < 5) {
        changed = kMeansLoop(flatWeights, centroids);
        kMeansCount++;
    }

    // if any empty species exist, we need to assign them
    bool[] nonEmpty = new bool[LEEAParams.SPECIESCOUNT];
	//for (int i = 0; i < genomeList.Count; i++)
    Parallel.For(0, genomeList.Count, po, i => {
        nonEmpty[genomeList[i].Species] = true;
    });

    for (int i = 0; i < nonEmpty.Length; i++)
        if (!nonEmpty[i]) {
            // find the genome that is furthest from its centroid
            double furthest = 0;
            int furthestIndex = 0;
            Object lockObj = new object();

			//for (int j = 0; j < genomeList.Count; j++)
            Parallel.For(0, genomeList.Count, po, j => {
                double distance = calculateVectorDistance(flatWeights[j], centroids[genomeList[j].Species]);

                if (distance > furthest) {
                    lock (lockObj) {
                        // now that we have lock, check again in case furthest was updated by another thread
                        if (distance > furthest) {
                            furthest = distance;
                            furthestIndex = j;
                        }
                    }
                }
            });

            // set this genome's weights as the new centroid
            Array.Copy(flatWeights[furthestIndex], centroids[i], flatWeights[furthestIndex].Length);
            genomeList[furthestIndex].Species = i;

            // now respeciate the population with this new centroid in place
            kMeansLoop(flatWeights, centroids, true);
        }
    }
