package algorithms;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Array;
import java.util.Arrays;
import weka.core.Instances;
import weka.core.pmml.FieldMetaInfo.Value;

public class Density extends Instances {

	double TotalCost;
    double Accuracy;
	/**
	 * ʵ����ȫ���ܶ�rho.
	 */
	double[] rho;
	/**
	 * rho������������������.
	 */
	int[] ordrho;
	/**
	 * ʵ������С����.
	 */
	double[] delta;
	/**
	 * ʵ����master.
	 */
	int[] master;
	/**
	 * ʵ�������ȼ�.
	 */
	double[] priority;
	/**
	 * ��������
	 */
	int[] centers;
	/**
	 * ����Ϣ����������һ�أ�
	 */
	int[] clusterIndices;
	/**
	 * �ܶ���ֵ.
	 */
	double dc;
	/**
	 * ������.
	 */
	double maximalDistance;
	/**
	 * ����Ϣ��.
	 */
	int[][] blockInformation;
	/**
	 * ����Ԥ������ǩ
	 */
	int[] predictedLabels;
	/**
	 * ���й���ı�ǩ��Ŀ
	 */
	int numTeach;
	/**
	 * ����Ԥ��ı�ǩ��Ŀ
	 */
	int numPredict;
	/**
	 * ����ͶƱ�ı�ǩ��Ŀ
	 */
	int numVote;
	/**
	 * ����ʵ��priority����
	 */
	int[] descendantIndices;
	/**
	 * ��ǩ����ָʾ
	 */
	boolean[] alreadyClassified;
	/**
	 ********************************** 
	 * ������
	 * 
	 * @throws IOException
	 ********************************** 
	 */

	public Density(Reader paraReader) throws IOException, Exception {
		super(paraReader);
	} // of the first constructor

	/**
	 ********************************** 
	 * 1. �ܶȾ���
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * 1.1. �������
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Manhattan distance.
	 ********************************** 
	 */
	public double manhattan(int paraI, int paraJ) {
		double tempDistance = 0;

		for (int i = 0; i < numAttributes() - 1; i++) {
			tempDistance += Math.abs(instance(paraI).value(i)
					- instance(paraJ).value(i));
		}// of for i

		return tempDistance;
	}// of manhattan

	/**
	 ********************************** 
	 * 1.2. �����ܶ�rho
	 ********************************** 
	 */
	/**
	 ********************************** 
	 * Compute rho
	 ********************************** 
	 */

	public void computeRho() {
		rho = new double[numInstances()];

		for (int i = 0; i < numInstances() - 1; i++) {
			for (int j = i + 1; j < numInstances(); j++) {
				if (manhattan(i, j) < dc) {
					rho[i] = rho[i] + 1;
					rho[j] = rho[j] + 1;
				}// of if
			}// of for j
		}// of for i

	}// of computeRho

	/**
	 ********************************** 
	 * Set dc.
	 ********************************** 
	 */
	public void setDc(double paraPercentage) {

		dc = maximalDistance * paraPercentage;
	}// of setDc

	/**
	 ********************************** 
	 * 1.3. ������С����delta
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Compute delta
	 ********************************** 
	 */

	public void computeDelta() {
		delta = new double[numInstances()];
		master = new int[numInstances()];
		ordrho = new int[numInstances()];

		// Step 1. rho����
		ordrho = mergeSortToIndices(rho);

		// Step 2. delta[ordrho[0]]
		delta[ordrho[0]] = maximalDistance;

		// Step 3. ����С����

		for (int i = 1; i < numInstances(); i++) {
			delta[ordrho[i]] = maximalDistance;
			for (int j = 0; j <= i - 1; j++) {
				if (manhattan(ordrho[i], ordrho[j]) < delta[ordrho[i]]) {
					delta[ordrho[i]] = manhattan(ordrho[i], ordrho[j]);
					master[ordrho[i]] = ordrho[j];
				}// of if
			}// of for j
		}// of for i

	}// of computeDelta

	/**
	 ********************************** 
	 * 1.4. �������ĵ�
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Compute centers.
	 ********************************** 
	 */

	// Step 1. �������ȼ� priority = rho*delta

	/**
	 ********************************** 
	 * Compute priority.
	 ********************************** 
	 */
	public void computePriority() {
		priority = new double[numInstances()];

		for (int i = 0; i < numInstances(); i++) {

			priority[i] = rho[i] * delta[i];

		}// of for i

	}// of computePriority

	// Step 2. ����priority���򣬴��ϵ�������ѡ��k������

	public void computeCenters(int paraNumbers) {
		centers = new int[paraNumbers];

		// Step 1. ��priority����
		int[] ordPriority = mergeSortToIndices(priority);

		// Step 2. ѡ��k������
		for (int i = 0; i < paraNumbers; i++) {
			centers[i] = ordPriority[i];
		}// of for i

	}// of computeCenters

	/**
	 ********************************** 
	 * 1.5. �������ĵ���о���
	 ********************************** 
	 */
	public void clusterWithCenters() {

		// Step 1. ��ʼ��
		int[] cl = new int[numInstances()];
		clusterIndices = new int[numInstances()];

		for (int i = 0; i < numInstances(); i++) {
			cl[i] = -1;
		}// of for i

		// Step 2. �����ĵ�������ǩ

		int tempNumber = 0;
		int tempCluster = 0;
		for (int i = 0; i < numInstances(); i++) {
			if (tempNumber < centers.length) {
				cl[centers[i]] = tempCluster;
				tempNumber++;
				tempCluster++;
			}// of if
		}// of for i

		/*// Step 2 ���ĸ����
				// System.out.println("this is the test 1" );
				for (int i = 0; i < centers.length; i++) {
					cl[centers[i]] = i;
				}// of for i
*/		
		
		// Step 3. �������������ǩ�����ǩ����masterһ�£�

		for (int i = 0; i < numInstances(); i++) {
			if (cl[ordrho[i]] == -1) {
				cl[ordrho[i]] = cl[master[ordrho[i]]];
			}// of if
		}// of for i

		// Step 4. �������Ϣ
		for (int i = 0; i < numInstances(); i++) {

			clusterIndices[i] = centers[cl[i]];

		}// of for i

	}// of clusterWithCenters

	/**
	 ********************************** 
	 * 2. ����ѧϰ
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * 2.1. ���ɿ���Ϣ��
	 ********************************** 
	 */
	/**
	 ********************************** 
	 * Compute block information
	 ********************************** 
	 */

	public void computeBlockInformation() {

		blockInformation = new int[centers.length][];

		for (int i = 0; i < centers.length; i++) {

			// Step 1. ͳ��ÿһ�����ж���ʵ��.
			int tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (clusterIndices[j] == centers[i]) {
					tempElements++;
				}// of if
			}// of for j

			// Step 2. ��ÿһ��Ԫ�ش������Ϣ��blockInformation.
			blockInformation[i] = new int[tempElements]; // ��Ҫ����ʼ����
			tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (clusterIndices[j] == centers[i]) {
					blockInformation[i][tempElements] = j;
					tempElements++;
				}// of if
			}// of for j

		}// of for i

	}// of computeBlockInformation

	/**
	 ********************************** 
	 * 2.2. ����ѡ������
	 ********************************** 
	 */

	// ���룺�ֿ�����ÿ�鹺��ĸ������ܵĹ������
	// ���������ʵ�������ǩ

	public void activeLearning(int paraBlock, int paraTeachEachBlock,
			int paraTeach) {

		// Step 1. ��ʼ��
		// ��ʼ������
		// ��Ҫ���� computeRho(); computeDelta(); computePriority();
		// ����rho, delta, priority
		predictedLabels = new int[numInstances()];

		for (int i = 0; i < numInstances(); i++) {
			predictedLabels[i] = -1;
		}// of for i
			// ��Ϊ����ͶƱ�׶���Ҫʶ����Щ��ǩδ�ܱ����

		numTeach = 0;
		numPredict = 0;
		numVote = 0;

		int tempBlocks = paraBlock;

		computeRho();
		computeDelta();
		computePriority();
		descendantIndices = mergeSortToIndices(priority);
		alreadyClassified = new boolean[numInstances()];

		// Step 2. ����ѧϰ��������
		// ��Ҫwhile ѭ��
		// �����˳���׼
		// 1) ����ʵ����ǽ���
		// 2�����ܹ�����ı�ǩ�����Ѿ�������

		while (true) {
			// {
			// Step 2.1 ����;
			computeCenters(tempBlocks);

			clusterWithCenters();

			computeBlockInformation();

			
			// Step 2.2 �ж���Щ���Ѿ�������
			boolean[] tempBlockProcessed = new boolean[tempBlocks];
			int tempUnProcessedBlocks = 0;
			for (int i = 0; i < blockInformation.length; i++) {
				tempBlockProcessed[i] = true;
				for (int j = 0; j < blockInformation[i].length; j++) {

					if (!alreadyClassified[blockInformation[i][j]]) {
						tempBlockProcessed[i] = false;
						tempUnProcessedBlocks++;
						break;
					}// of if
				}// of for j
			}// of for i
			
			// Step 2.3 �����ǩ������ѧϰ
			// �����ǩ�������¼������

			for (int i = 0; i < blockInformation.length; i++) {
				// Step 2.3.1 ����ÿ��Ѿ��������꣬��ֱ���˳�������Ҫ����

				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				// Step 2.3.2 ���ĳһ��̫С������δ�ܱ��ʵ������С��ÿ�鹺����������ȫ������
				
				if (blockInformation[i].length < paraTeachEachBlock) {
					
					for (int j = 0; j < blockInformation[i].length; j++) {
						if (!alreadyClassified[blockInformation[i][j]]) {
							if (numTeach >= paraTeach) {
								break;
							}// of if
							predictedLabels[blockInformation[i][j]] = (int) instance(
									blockInformation[i][j]).classValue();
							alreadyClassified[blockInformation[i][j]] = true;
							numTeach++;
							System.out.println("numTeach first = " + numTeach);
						}// of if
					}// of for j
				}// of if
				
				// Step 2.3.3 ʣ�µĿ飬�������ʵ��������ÿ���ܹ���ı�ǩ��
				double[] tempPriority = new double[blockInformation[i].length];
				int[] ordPriority = new int[blockInformation[i].length];

				int tempIndex = 0;
				for (int j = 0; j < numInstances(); j++) {
					if (clusterIndices[descendantIndices[j]] == centers[i]) {
						ordPriority[tempIndex] = descendantIndices[j];
						// tempPriority[tempIndex] = priority[j];
						tempIndex++;
					}// of if
				}// of for j

				int tempNumTeach = 0;
				for (int j = 0; j < blockInformation[i].length; j++) {
					if (alreadyClassified[ordPriority[j]]) {
						continue;
					}// of if
					if (numTeach >= paraTeach) {
						break;
					}// of if
					predictedLabels[ordPriority[j]] = (int) instance(
							ordPriority[j]).classValue();
					alreadyClassified[ordPriority[j]] = true;
					numTeach++;
					
					tempNumTeach++;
					
					if (tempNumTeach >= paraTeachEachBlock) {
						break;
					}// of if

				}// of for j

			} // of for i
			

			// Step 2.4 ��ʣ�µ�ʵ������Ԥ�⣬����

			boolean tempPure = true;

			for (int i = 0; i < blockInformation.length; i++) {

				// Step 2.4.1 �жϸÿ��Ƿ��Ѿ�����

				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				// Step 2.4.2 �ж�ĳ���Ƿ�Ϊ���Ŀ�

				boolean tempFirstLable = true;
				// ����һ����ǣ��ж�ǰ������ʵ���ı���Ƿ�һ��
				int tempCurrentInstance;
				int tempLable = 0;

				for (int j = 0; j < blockInformation[i].length; j++) {
					tempCurrentInstance = blockInformation[i][j];
					if (alreadyClassified[tempCurrentInstance]) {

						if (tempFirstLable) {
							tempLable = predictedLabels[tempCurrentInstance];
							tempFirstLable = false;
						} else {
							if (tempLable != predictedLabels[tempCurrentInstance]) {
								tempPure = false;
								break;
							}// of if
						} // of if
					}// of if
				}// of for j

				// Step 2.4.3 ����ÿ��Ǵ��ģ�ֱ�ӷ�������ʣ�µ�ʵ���������ͬ�ı�ǩ
				if (tempPure) {
					for (int j = 0; j < blockInformation[i].length; j++) {
						if (!alreadyClassified[blockInformation[i][j]]) {
							predictedLabels[blockInformation[i][j]] = tempLable;
							alreadyClassified[blockInformation[i][j]] = true;
							numPredict++;
						}// of if
					}// of for j
				}// of if
			}// of for i

			// Step 2.4.4 ����ÿ鲻�Ǵ��ģ������¾��࣬�������ѳɸ�С�Ŀ�

			tempBlocks++;

			// Step 2.5 �˳�

			// Step 2.5.1 ������еĿ鶼�Ѿ����������˳�
			if (tempUnProcessedBlocks == 0) {
				break;
			}// of if

			// Step 2.5.2 ������й���ı�ǩ�Ѿ����꣬��ֱ���˳�ѭ��
			if (numTeach >= paraTeach) {
				break;
			}// of if

		} // of while

		// Step 3 ��������Ѿ������ǩ��ָ�����꣬��Ȼ��δ�ܷ����ʵ������ͶƱ����ʣ��ʵ���ı�ǩ

		// Step 3.1 ������Ԥ���ǣ����·���
		int max = getMax(predictedLabels);
		computeCenters(max + 1);
		clusterWithCenters();
		computeBlockInformation();
		// Step 3.2 ͶƱ

		int[][] vote = new int[max + 1][max + 1];
		int voteIndex = -1;

		for (int i = 0; i < blockInformation.length; i++) {

			// Step 3.2.1 ͳ�Ʊ�ǩ
			for (int j = 0; j < blockInformation[i].length; j++) {

				for (int k = 0; k <= max; k++) {

					if (predictedLabels[blockInformation[i][j]] == k) {
						vote[i][k]++;
					}// of if
				}// of for k
			}// of for j

			// Step 3.2.2 ����ÿһ����ʵ�����ķ���
			voteIndex = getMaxIndex(vote[i]);

			// Step 5.2.3 ����һ��������δ�����ǻ��������ı��

			for (int j = 0; j < blockInformation[i].length; j++) {
				if (predictedLabels[blockInformation[i][j]] == -1) {
					predictedLabels[blockInformation[i][j]] = voteIndex;
					numVote++;
				}// of if
			}// of for j
		}// of for i

		System.out.println("clusterBasedActiveLearning finish!");
		System.out.println("numTeach = " + numTeach + "; predicted = "
				+ numPredict + "; numVote = " + numVote);
		System.out.println("Accuracy = " + getPredictionAccuracy());

	}// of activeLearing

	/**
	 ********************************** 
	 * �����׼ȷ��
	 ********************************** 
	 */

	public double getPredictionAccuracy() {
		double tempIncorrect = 0;
		double tempAccuracy = 0;

		for (int i = 0; i < numInstances(); i++) {
			if (predictedLabels[i] != (int) instance(i).classValue()) {
				tempIncorrect++;
			}// of if
		}// of for i

		tempAccuracy = (numInstances() - tempIncorrect - numTeach)
				/ (numInstances() - numTeach);

		System.out.println("Incorrected = " + tempIncorrect);
		return tempAccuracy;
	}// of getPredictionAccuracy

	/**
	 ********************************** 
	 * ��������ĺ���
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Compute maximal distance.
	 ********************************** 
	 */

	public void computeMaximalDistance() {
		// maximalDistance = 0;
		double tempDistance;
		for (int i = 0; i < numInstances(); i++) {
			for (int j = 0; j < numInstances(); j++) {
				tempDistance = manhattan(i, j);
				if (maximalDistance < tempDistance) {
					maximalDistance = tempDistance;
				}// of if
			}// of for j
		}// of for i

	}// of computeMaximalDistance

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged.<br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2].<br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].
	 * 
	 * @author Fan Min 2016/09/09
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */

	public static  int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];//����ά�Ƚ����洢����tempIndex����
		
		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i
			// System.out.println("Initialize, resultMatrix = " +
			// Arrays.deepToString(resultMatrix));

		// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;
		while (tempCurrentLength < tempLength) {
			// System.out.println("tempCurrentLength = " + tempCurrentLength);
			// Divide into a number of groups
			// Here the boundary is adaptive to array length not equal to 2^k.
			// ceil������ȡ������
			
			for (int i = 0; i < Math.ceil(tempLength + 0.0 / tempCurrentLength) / 2; i++) {//��λ����һ��
				// Boundaries of the group
				tempFirstStart = i * tempCurrentLength * 2;
				//tempSecondStart��λ�ڶ��鿪ʼ��λ��index
				tempSecondStart = tempFirstStart + tempCurrentLength;//���������ж��Ƿ������һС�飬������ʼ���Ĺ���
//				if (tempSecondStart >= tempLength) {
//					
//					break;
//				} // Of if
				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {  //�������һС�顣�����������峤�ȣ���tempSecondEnd��λ���������
					tempSecondEnd = tempLength - 1;
				} // Of if
//					 System.out.println("tempFirstStart = " + tempFirstStart +
//					 ", tempSecondStart = " + tempSecondStart
//					 + ", tempSecondEnd = " + tempSecondEnd);

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;
				// System.out.println("Before merge");
				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;						
					} // Of for j
				break;
			} // Of if
				
				while ((tempFirstIndex <= tempSecondStart - 1) && (tempSecondIndex <= tempSecondEnd)) {//������ʼ������Ĺ���
					
					if (paraArray[resultMatrix[tempIndex % 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex
							% 2][tempSecondIndex]]) {
						//System.out.println("tempIndex + 1) % 2"+tempIndex);
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempFirstIndex];
						int a =(tempIndex + 1) % 2;
						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempSecondIndex];
						int b =(tempIndex + 1) % 2;
						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;
				   
				} // Of while
					// System.out.println("After compared merge");
				// Remaining part
				// System.out.println("Copying the remaining part");
				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];
					tempCurrentIndex++;
					
				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex % 2][j];					
					tempCurrentIndex++;
				} // Of for j
				//paraArray=resultMatrix[0];
					// System.out.println("After copying remaining part");
				// System.out.println("Round " + tempIndex + ", resultMatrix = "
				// + Arrays.deepToString(resultMatrix));
			} // Of for i
				// System.out.println("Round " + tempIndex + ", resultMatrix = "
				// + Arrays.deepToString(resultMatrix));

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices
	

	/**
	 ********************************** 
	 * �����ֵ.
	 ********************************** 
	 */
	public int getMax(int[] paraArray) {
		int max = paraArray[0];
		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] > max) {
				max = paraArray[i];
			}// of if
		}// of for i

		return max;
	}// of getMax

	/**
	 ********************************** 
	 * �����ֵ���ڵ�λ��.
	 ********************************** 
	 */
	public int getMaxIndex(int[] paraArray) {
		int maxIndex = 0;
		int tempIndex = 0;
		int max = paraArray[0];

		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] > max) {
				max = paraArray[i];
				tempIndex = i;
			}// of if
		}// of for i
		maxIndex = tempIndex;
		return maxIndex;
	}// of getMaxIndex

	
	/**
	 ********************************** 
	 * ������ܴ��ۣ�����������ͬ��CSAL 
	 ********************************** 
	 */

	public void getPredictionTotalCost() {
		double tempIncorrect = 0;
		TotalCost = 0;
        Accuracy =  0;
		for (int i = 0; i < numInstances(); i++) {
			if (predictedLabels[i] != (int) instance(i).classValue()) {
				tempIncorrect++;
			}// of if
		}// of for i

	Accuracy = (numInstances() - tempIncorrect - numTeach)
				/ (numInstances() - numTeach);

		TotalCost = tempIncorrect * 20 + numTeach * 10;
		System.out.println("CSALIncorrect" + tempIncorrect);
		
	}// of getPredictionAccuracy
	
	/**
	 ******************* 
	 * ���Ժ���
	 ******************* 
	 */

	public static void Test() {

		String arffFilename = "D:/data/iris.arff";

		try {
			FileReader fileReader = new FileReader(arffFilename);
			Density tempData = new Density(fileReader);
			fileReader.close();
			tempData.setClassIndex(tempData.numAttributes() - 1);
			tempData.computeMaximalDistance();
			tempData.setDc(0.1);
			tempData.activeLearning(2, 2, 15);
			System.out.println("predictedLabels"
					+ Arrays.toString(tempData.predictedLabels));
			tempData.getPredictionTotalCost();			
			System.out.println("Accuracy" + tempData.Accuracy);

		} catch (Exception ee) {
		}// of try

	}// of densityTest

	/**
	 ********************************** 
	 * ������
	 ********************************** 
	 */
	public static void main(String args[]) {
		System.out.println("Hello, density");
		Test();
	}// of main

}

