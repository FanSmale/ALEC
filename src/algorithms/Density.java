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
	 * 实例的全局密度rho.
	 */
	double[] rho;
	/**
	 * rho排序后的数组索引数组.
	 */
	int[] ordrho;
	/**
	 * 实例的最小距离.
	 */
	double[] delta;
	/**
	 * 实例的master.
	 */
	int[] master;
	/**
	 * 实例的优先级.
	 */
	double[] priority;
	/**
	 * 聚类中心
	 */
	int[] centers;
	/**
	 * 簇信息，我属于哪一簇？
	 */
	int[] clusterIndices;
	/**
	 * 密度阈值.
	 */
	double dc;
	/**
	 * 最大距离.
	 */
	double maximalDistance;
	/**
	 * 块信息表.
	 */
	int[][] blockInformation;
	/**
	 * 所有预测的类标签
	 */
	int[] predictedLabels;
	/**
	 * 所有购买的标签数目
	 */
	int numTeach;
	/**
	 * 所有预测的标签数目
	 */
	int numPredict;
	/**
	 * 所有投票的标签数目
	 */
	int numVote;
	/**
	 * 所有实例priority排序
	 */
	int[] descendantIndices;
	/**
	 * 标签分配指示
	 */
	boolean[] alreadyClassified;
	/**
	 ********************************** 
	 * 构造体
	 * 
	 * @throws IOException
	 ********************************** 
	 */

	public Density(Reader paraReader) throws IOException, Exception {
		super(paraReader);
	} // of the first constructor

	/**
	 ********************************** 
	 * 1. 密度聚类
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * 1.1. 计算距离
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
	 * 1.2. 计算密度rho
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
	 * 1.3. 计算最小距离delta
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

		// Step 1. rho排序
		ordrho = mergeSortToIndices(rho);

		// Step 2. delta[ordrho[0]]
		delta[ordrho[0]] = maximalDistance;

		// Step 3. 找最小距离

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
	 * 1.4. 计算中心点
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * Compute centers.
	 ********************************** 
	 */

	// Step 1. 计算优先级 priority = rho*delta

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

	// Step 2. 根据priority排序，从上到下依次选择k个中心

	public void computeCenters(int paraNumbers) {
		centers = new int[paraNumbers];

		// Step 1. 对priority排序
		int[] ordPriority = mergeSortToIndices(priority);

		// Step 2. 选择k个中心
		for (int i = 0; i < paraNumbers; i++) {
			centers[i] = ordPriority[i];
		}// of for i

	}// of computeCenters

	/**
	 ********************************** 
	 * 1.5. 根据中心点进行聚类
	 ********************************** 
	 */
	public void clusterWithCenters() {

		// Step 1. 初始化
		int[] cl = new int[numInstances()];
		clusterIndices = new int[numInstances()];

		for (int i = 0; i < numInstances(); i++) {
			cl[i] = -1;
		}// of for i

		// Step 2. 给中心点分配类标签

		int tempNumber = 0;
		int tempCluster = 0;
		for (int i = 0; i < numInstances(); i++) {
			if (tempNumber < centers.length) {
				cl[centers[i]] = tempCluster;
				tempNumber++;
				tempCluster++;
			}// of if
		}// of for i

		/*// Step 2 中心给标记
				// System.out.println("this is the test 1" );
				for (int i = 0; i < centers.length; i++) {
					cl[centers[i]] = i;
				}// of for i
*/		
		
		// Step 3. 给其余点分类类标签（类标签与其master一致）

		for (int i = 0; i < numInstances(); i++) {
			if (cl[ordrho[i]] == -1) {
				cl[ordrho[i]] = cl[master[ordrho[i]]];
			}// of if
		}// of for i

		// Step 4. 分配簇信息
		for (int i = 0; i < numInstances(); i++) {

			clusterIndices[i] = centers[cl[i]];

		}// of for i

	}// of clusterWithCenters

	/**
	 ********************************** 
	 * 2. 主动学习
	 ********************************** 
	 */

	/**
	 ********************************** 
	 * 2.1. 生成块信息表
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

			// Step 1. 统计每一簇中有多少实例.
			int tempElements = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (clusterIndices[j] == centers[i]) {
					tempElements++;
				}// of if
			}// of for j

			// Step 2. 将每一个元素存入块信息表blockInformation.
			blockInformation[i] = new int[tempElements]; // 重要：初始化行
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
	 * 2.2. 主动选择样本
	 ********************************** 
	 */

	// 输入：分块数，每块购买的个数，总的购买个数
	// 输出：所有实例的类标签

	public void activeLearning(int paraBlock, int paraTeachEachBlock,
			int paraTeach) {

		// Step 1. 初始化
		// 初始化变量
		// 主要包括 computeRho(); computeDelta(); computePriority();
		// 计算rho, delta, priority
		predictedLabels = new int[numInstances()];

		for (int i = 0; i < numInstances(); i++) {
			predictedLabels[i] = -1;
		}// of for i
			// 因为后续投票阶段需要识别哪些标签未能被标记

		numTeach = 0;
		numPredict = 0;
		numVote = 0;

		int tempBlocks = paraBlock;

		computeRho();
		computeDelta();
		computePriority();
		descendantIndices = mergeSortToIndices(priority);
		alreadyClassified = new boolean[numInstances()];

		// Step 2. 主动学习迭代过程
		// 主要while 循环
		// 两种退出标准
		// 1) 所有实例标记结束
		// 2）所能够购买的标签数量已经结束。

		while (true) {
			// {
			// Step 2.1 聚类;
			computeCenters(tempBlocks);

			clusterWithCenters();

			computeBlockInformation();

			
			// Step 2.2 判断哪些块已经被处理
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
			
			// Step 2.3 购买标签，主动学习
			// 购买标签，分以下几种情况

			for (int i = 0; i < blockInformation.length; i++) {
				// Step 2.3.1 如果该块已经被处理完，则直接退出，不需要购买

				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				// Step 2.3.2 如果某一块太小，所有未能标记实例个数小于每块购买总数，则全部购买
				
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
				
				// Step 2.3.3 剩下的块，所需分类实例数大于每块能购买的标签数
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
			

			// Step 2.4 对剩下的实例进行预测，分类

			boolean tempPure = true;

			for (int i = 0; i < blockInformation.length; i++) {

				// Step 2.4.1 判断该块是否已经分类

				if (tempBlockProcessed[i]) {
					continue;
				}// of if

				// Step 2.4.2 判断某块是否为纯的块

				boolean tempFirstLable = true;
				// 设置一个标记，判断前后两个实例的标记是否一致
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

				// Step 2.4.3 如果该块是纯的，直接分类所有剩下的实例，获得相同的标签
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

			// Step 2.4.4 如果该块不是纯的，则重新聚类，继续分裂成更小的块

			tempBlocks++;

			// Step 2.5 退出

			// Step 2.5.1 如果所有的块都已经被处理，则退出
			if (tempUnProcessedBlocks == 0) {
				break;
			}// of if

			// Step 2.5.2 如果所有购买的标签已经用完，则直接退出循环
			if (numTeach >= paraTeach) {
				break;
			}// of if

		} // of while

		// Step 3 如果所有已经购买标签的指标用完，仍然有未能分类的实例，则投票决定剩下实例的标签

		// Step 3.1 根据已预测标记，重新分类
		int max = getMax(predictedLabels);
		computeCenters(max + 1);
		clusterWithCenters();
		computeBlockInformation();
		// Step 3.2 投票

		int[][] vote = new int[max + 1][max + 1];
		int voteIndex = -1;

		for (int i = 0; i < blockInformation.length; i++) {

			// Step 3.2.1 统计标签
			for (int j = 0; j < blockInformation[i].length; j++) {

				for (int k = 0; k <= max; k++) {

					if (predictedLabels[blockInformation[i][j]] == k) {
						vote[i][k]++;
					}// of if
				}// of for k
			}// of for j

			// Step 3.2.2 计算每一块中实例最多的分类
			voteIndex = getMaxIndex(vote[i]);

			// Step 5.2.3 将这一块中所有未分类标记获得最多分类的标记

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
	 * 求分类准确度
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
	 * 其他所需的函数
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
		int[][] resultMatrix = new int[2][tempLength];//两个维度交换存储排序tempIndex控制
		
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
			// ceil是向上取整函数
			
			for (int i = 0; i < Math.ceil(tempLength + 0.0 / tempCurrentLength) / 2; i++) {//定位到哪一块
				// Boundaries of the group
				tempFirstStart = i * tempCurrentLength * 2;
				//tempSecondStart定位第二块开始的位置index
				tempSecondStart = tempFirstStart + tempCurrentLength;//可以用于判断是否是最后一小块，并做初始化的工作
//				if (tempSecondStart >= tempLength) {
//					
//					break;
//				} // Of if
				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {  //控制最后一小块。若超过了整体长度，则当tempSecondEnd定位到数组最后
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
				
				while ((tempFirstIndex <= tempSecondStart - 1) && (tempSecondIndex <= tempSecondEnd)) {//真正开始做排序的工作
					
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
	 * 求最大值.
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
	 * 求最大值所在的位置.
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
	 * 求分类总代价（误分类代价相同）CSAL 
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
	 * 测试函数
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
	 * 主函数
	 ********************************** 
	 */
	public static void main(String args[]) {
		System.out.println("Hello, density");
		Test();
	}// of main

}

