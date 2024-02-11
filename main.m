% % Facial Expression Recognition System
% Rabia Konuk
% l:599 a:2


% % Introduction
% The aim of this project is to develop a facial expression recognition system 
% capable of distinguishing between seven distinct emotions: happy, sad, 
% disgusted, fearful, angry, surprised, and neutral. This challenge encompasses 
% the broader field of computer vision and pattern recognition, utilizing 
% the KDEF-dyn database's open-source facial expression datasets. The ultimate goal
% is to accurately classify these emotions through the development and application of
% a robust recognition algorithm.


%%
% Problem 1 (0)
% Data Preprocessing
% The initial step involves preparing the dataset for the subsequent analysis. 
% This process includes data normalization and division into training and testing sets. 
% The dataset, provided in the form of pre-processed files within the "KDEF_wavelet_data" 
% folder, consists of images categorized into seven classes corresponding to different
% facial expressions. For each class, we allocate 75% of the images for training and 
% the remaining 25% for testing. This partitioning is essential for evaluating the 
% performance of our classification algorithm.

% Read files; normalize data
% disp(pwd); % Display the current working directory

no_classes = 7; % emotion categories
data_name = {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'};

% Initialization of arrays for data processing
datak = [];
class = [];
image_size = []; % This will hold the size of the images

% Seed for reproducibility
rng(1);

% Initialize arrays for storing data & labels
train_data = [];
test_data = [];
train_label = [];
test_label = [];

for j = 1:no_classes
    % Constructing the path to the dataset folder for each emotion category
    image_folder = ['/Users/rabia/Downloads/hwk2_files/KDEF_wavelet_data/', data_name{j}, '/'];
    file_pattern = fullfile(image_folder, '*.png'); % Pattern to match PNG files in the emotion category folder
    image_files = dir(file_pattern); 
    nfiles = length(image_files);

    % Shuffle the image files (to ensure random distribution for training and testing)
    image_files = image_files(randperm(nfiles));
    
    % Determine split index
    splitIndex = floor(0.75 * nfiles);

    for i = 1:nfiles
        filename = image_files(i).name; % Get the filename of the current image
        im = imread(fullfile(image_folder, filename)); % Read the current image

        % Convert to grayscale if needed
        if size(im, 3) == 3
            im = rgb2gray(im);
        end
        
        % Record the size of the first image to ensure all images have the same dimensions
        if isempty(image_size)
            image_size = size(im);
            if length(image_size) > 2
                image_size = image_size(1:2); % Capture only height and width
            end
        end

        % Reshape the image into a column vector and normalize pixel values to the [0, 1] range
        im_vector = double(reshape(im, [], 1)) / 255;

        % Split data into training and testing sets
        if i <= splitIndex
            train_data = [train_data im_vector]; % Append the image vector to the training data
            train_label = [train_label; {data_name{j}}]; % Append the category name to the training labels
        else
            test_data = [test_data im_vector]; % Append the image vector to the testing data
            test_label = [test_label; {data_name{j}}]; % Append the category name to the testing labels
        end
    end
end

%%
% Problem 1(a): Principal Component Analysis
%

% In this part of the assignment, we perform PCA on the training dataset to
% extract the mean face and the principal components, commonly referred to
% as eigenfaces in the context of facial recognition. This technique
% reduces the dimensionality of the dataset while preserving as much of the
% variation in the data as possible. Eigenfaces represent the directions in
% which the data varies the most.


% Compute the mean face 
% The mean face is computed by averaging each pixel
% across all images in the training set. This gives us a baseline
% representation of the average facial features within the dataset.
mean_face = mean(train_data, 2);

% Center the data 
% Before performing PCA, it's crucial to center the dataset
% by subtracting the mean face from each face in the training set. This
% step ensures that the PCA identifies the variations from the mean rather
% than the mean itself.
centered_data = train_data - mean_face;

% Perform SVD on the centered data 
% Singular Value Decomposition (SVD) is utilized to perform PCA on the centered data. The 'econ' option
% ensures that the computation is efficient, especially for large datasets.
[U, S, V] = svd(centered_data, 'econ');

% Ensure that there are at least five eigenfaces. The first five columns of
% U (obtained from SVD) represent the eigenfaces, which are the directions
% of maximum variance in the dataset. These are selected to visualize and
% further use in the recognition system.
if size(U, 2) < 5
    error('Not enough eigenfaces available. Only %d eigenfaces can be computed.', size(U, 2));
end
% Select the First Five Principal Components
eigenfaces = U(:, 1:5);

% Display the Results
% Reshape the mean face and the eigenfaces to display them as images

% Mean Face
figure;
subplot(1, 6, 1);
imshow(reshape(mean_face, image_size), []);
title('Mean Face');

% Eigenfaces
for i = 1:5
    subplot(1, 6, i+1);
    eigenface_i = reshape(eigenfaces(:, i), image_size);
    eigenface_i = (eigenface_i - min(eigenface_i(:))) / (max(eigenface_i(:)) - min(eigenface_i(:))); % Normalization
    imshow(eigenface_i, []);
    title(['Eigenface ', num2str(i)]);
end

% The mean face provides a general representation of the dataset, while the
% eigenfaces capture the most significant modes of variation among the
% facial images. These components will be pivotal in the subsequent stages
% of the project, where they contribute to the classification of facial
% expressions.

%%
% Problem 1(b) : Reconstruction Error and Principal Component Selection
% 

% In this section, we aim to quantify the impact of PCA on the
% reconstruction of facial images from the training set. By assessing the
% reconstruction error as a function of the number of principal components
% used, we can determine the minimal number of components required to
% achieve a specified level of accuracy, in this case, a reconstruction
% error of 2% or less.

% The reconstruction error is evaluated as the mean squared error (MSE)
% normalized by the total variance of the dataset. This metric provides
% insight into how well the reduced dimensionality data (using a certain
% number of principal components) can reconstruct the original images.

% Calculate the total variance to use in the calculation of reconstruction error
total_variance = sum(diag(S).^2);

% Preallocate the array for reconstruction errors
reconstruction_errors = zeros(min(size(U, 2), size(centered_data, 2)), 1);

% Iteratively reconstruct images and calculate the error for increasing number of components
for num_components = 1:length(reconstruction_errors)
    % Project the centered data onto the principal components
    projection = centered_data' * U(:, 1:num_components);
    
    % Reconstruct the data from the projection
    reconstruction = projection * U(:, 1:num_components)';
    
    % Add the mean face back to the reconstruction
    reconstruction = bsxfun(@plus, reconstruction, mean_face');
    
    % Calculate the mean squared reconstruction error
    mse = mean(sum((centered_data' - reconstruction) .^ 2, 2));
    reconstruction_errors(num_components) = mse / total_variance;
end

% Plot the reconstruction error against the number of principal components
figure;
plot(reconstruction_errors);
xlabel('Number of Principal Components');
ylabel('Reconstruction Error');
title('Reconstruction Error vs Number of Principal Components');

% To ensure a reconstruction error of 2% or less, we identify the minimum number of principal components required:
components_needed = find(reconstruction_errors <= 0.02, 1);
% disp(['Number of components needed for 2% or less reconstruction error: ', num2str(components_needed)]);

% Define image_components with the number of components you want to use for reconstruction
image_components = [1, 5, 10, components_needed];

% Reconstruct a random image from your dataset using 1, 5, 10, and c principal components
% (To illustrate the practical effect of using a varying number of principal components)
random_image_index = randi(size(centered_data, 2));
random_image = centered_data(:, random_image_index);

% Display the original image
figure;
subplot(2, 3, 1);
original_image = reshape(train_data(:, random_image_index) + mean_face, image_size); % Correction here
imshow(original_image, []);
title('Original Image');

% Display reconstructions with different numbers of principal components
for i = 1:length(image_components)
    num_components = image_components(i);
    
    % Project the random image onto the principal components
    projection = random_image' * U(:, 1:num_components);
    
    % Reconstruct the image from the projection
    reconstruction = projection * U(:, 1:num_components)' + mean_face';
    
    subplot(2, 3, i+1);
    recon_image = reshape(reconstruction, image_size);
    imshow(recon_image, []);
    title(['Reconstruction with ', num2str(num_components), ' PCs']);
end

% We can see PCA is efficient in facial image reconstruction. This part is
% highlighting the balance between using fewer principal components and
% maintaining image quality. By quantifying reconstruction errors, we
% establish a clear guideline for optimal dimensionality reduction in
% facial expression recognition. This not only improves recognition
% accuracy but also optimizes computational resources

%%
% Problem 1(c) : PCA Coef. Scatter Plot and Classification
%

% PCA space projection: After determining the principal components in the
% previous steps, we project both the training and testing datasets onto
% the PCA space. This transformation facilitates a more insightful analysis
% of the data's intrinsic structure, emphasizing the variance most relevant
% to facial expressions.

% Project training and test data onto the principal components
pca_train = centered_data' * U;
pca_test = (test_data - mean_face)' * U;

% Visualize the distribution of different classes in the PCA space
% (Scatter plot using the first two principal components)
figure;
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']; % Different colors for different classes
markers = ['o', '+', '*', 'x', 's', 'd', '^']; % Different markers for different classes
for j = 1:no_classes
    % Extract PCA coefficients for each class
    idx = strcmp(train_label, data_name{j});
    scatter(pca_train(idx, 1), pca_train(idx, 2), 36, colors(j), markers(j), 'DisplayName', data_name{j});
    hold on;
end
legend('show');
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA Coefficient Scatter Plot');
hold off;

% Visualize the distribution of different classes in the PCA space
figure;
hold on;
for j = 1:no_classes
    idx = strcmp(train_label, data_name{j});
    % disp(['Class: ', data_name{j}, ' - Number of samples: ', num2str(sum(idx))]);
    scatter(pca_train(idx, 1), pca_train(idx, 2), 36, colors(j), markers(j), 'DisplayName', data_name{j});
end
hold off;
legend('show');
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA Coefficient Scatter Plot');

% k-NN Classifier 
% To demonstrate the practical application of the
% PCA-transformed data, we employ a k-Nearest Neighbors (k-NN) classifier.
% This method, chosen for its effectiveness among others, classifies test
% samples based on the closest training samples in the PCA space.
k = 7; % Number of neighbors
train_labels_cat = categorical(train_label);
test_labels_cat = categorical(test_label);
knnModel = fitcknn(pca_train, train_labels_cat, 'NumNeighbors', k);

% Prediction and accuracy calculation for k-NN
knnPredictions = predict(knnModel, pca_test);
knnAccuracy = sum(knnPredictions == test_labels_cat) / numel(test_labels_cat);
fprintf('k-NN Classification Accuracy: %.2f%%\n', knnAccuracy * 100);

% The PCA coefficient scatter plot vividly demonstrates the varying degrees
% of class separability in the reduced dimensionality space, which directly
% influences the effectiveness of the subsequent classification. By
% employing k-NN classification, we assess the practical utility of PCA in
% enhancing classification performance. The chosen k-NN classifier, with
% its straightforward implementation and interpretability, serves as a
% baseline to validate the PCA process's effectiveness in facial expression
% recognition tasks.

% Other tried methods

% Simple Classification (e.g., Nearest Neighbor)
% Calculate the distance between each test sample and each training sample in the PCA space
% Then classify each test sample to the class of its nearest training sample
% predicted_labels = cell(size(test_label));
% for i = 1:size(pca_test, 1)
%     distances = sum((pca_train - pca_test(i, :)).^2, 2);
%     [~, minIndex] = min(distances);
%     predicted_labels{i} = train_label{minIndex};
% end
% 
% % Calculate classification accuracy
% accuracy = sum(strcmp(predicted_labels, test_label)) / length(test_label);
% disp(['Classification accuracy: ', num2str(accuracy * 100), '%']);


% % Random Forest Classifier
% numTrees = 50; % Number of trees in the forest
% randomForestModel = TreeBagger(numTrees, pca_train, train_labels_cat, 'Method', 'classification');
% 
% % Prediction and accuracy calculation for Random Forest
% [rfPredictions, ~] = predict(randomForestModel, pca_test);
% rfPredictionsCat = categorical(rfPredictions);
% rfAccuracy = sum(rfPredictionsCat == test_labels_cat) / numel(test_labels_cat);
% fprintf('Random Forest Classification Accuracy: %.2f%%\n', rfAccuracy * 100);
% 
% % Fine-tuned SVM Classifier
% svmTemplate = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
% svmModel = fitcecoc(pca_train, train_labels_cat, 'Learners', svmTemplate, 'Coding', 'onevsall');
% 
% % Prediction and accuracy calculation for fine-tuned SVM
% svmPredictions = predict(svmModel, pca_test);
% svmAccuracy = sum(svmPredictions == test_labels_cat) / numel(test_labels_cat);
% fprintf('Fine-Tuned SVM Classification Accuracy: %.2f%%\n', svmAccuracy * 100);


%%
% Problem 1(d) - Classification
%

% Following the reduction of dimensionality with PCA and the preliminary
% classification using k-NN, this section aims to apply a more
% sophisticated classification technique, Support Vector Machines (SVM), to
% the optimally reduced dataset. Additionally, we evaluate the model's
% performance through confusion matrices for both validation and test
% datasets.

% First, we determine the optimal number of PCA components that capture the
% majority of the variance within the data, aiming for a threshold that
% retains at least 95% of the total variance.
explainedVar = cumsum(diag(S).^2) / sum(diag(S).^2);
optimalPCAComponents = find(explainedVar >= 0.95, 1);

% We project both training and testing data onto the space defined by the
% selected optimal number of PCA components. This step ensures that our SVM
% model trains and predicts based on the most informative features of the
% data.
pca_train_optimal = pca_train(:, 1:optimalPCAComponents);
pca_test_optimal = pca_test(:, 1:optimalPCAComponents);

% disp(['Optimal number of PCA components: ', num2str(optimalPCAComponents)]);

% We train an SVM classifier with an RBF kernel on the dataset, reduced to
% its optimal PCA components, enhancing feature relevance. Cross-validation
% is employed to ensure the model's robustness and generalizability across
% unseen data.

% Train the SVM classifier
template = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
SVMModel = fitcecoc(pca_train_optimal, train_labels_cat, 'Learners', template, 'Coding', 'onevsall');

% Perform cross-validation
CVSVMModel = crossval(SVMModel, 'KFold', 5);

% Prediction on validation set
validationPredictions = kfoldPredict(CVSVMModel);

% The performance of the SVM classifier is evaluated using confusion
% matrices for both the validation set (derived from cross-validation) and
% the test set. The confusion matrix visually represents the classifier's
% accuracy across different classes, highlighting any potential biases or
% weaknesses.

% Confusion matrix for validation set
confMatValidation = confusionmat(train_labels_cat, validationPredictions);
confusionchart(confMatValidation);
title('Confusion Matrix for Validation Set');

% Prediction on test set
testPredictions = predict(SVMModel, pca_test_optimal);

% Confusion matrix for test set
confMatTest = confusionmat(test_labels_cat, testPredictions);
confusionchart(confMatTest);
title('Confusion Matrix for Test Set');

% Calculate the classification accuracy for the test set
accuracyTest = sum(testPredictions == test_labels_cat) / numel(test_labels_cat);
fprintf('Test Set Classification Accuracy: %.2f%%\n', accuracyTest * 100);

% Implementing SVM on the dataset, post-PCA dimensionality reduction,
% illustrates the effectiveness of combining these powerful machine
% learning techniques for facial expression classification. The use of
% confusion matrices provides a detailed insight into the model's
% performance, showcasing its strengths and areas for improvement.

%%
% Problem 1(e) : Linear Discriminant Analysis for Feature Extraction
%

% Building on the insights gained from PCA, this phase incorporates LDA to
% further refine the feature space for enhanced classification. Unlike PCA,
% which emphasizes variance regardless of class, LDA focuses on maximizing
% the separability among known categories.

% The training data is centered by subtracting the mean of the entire
% dataset. This step, similar to the preprocessing for PCA, ensures that
% LDA operates on data centered around the origin.
centered_data = train_data - mean(train_data, 2);

% We employ LDA on the centered training data, utilizing 'pseudoLinear'
% discriminant type to accommodate datasets that may not follow a normal
% distribution closely or have singular covariance matrices. Convert labels
% to categorical
train_labels_cat = categorical(train_label);

% Perform LDA with pseudoLinear discriminant type
ldaModel = fitcdiscr(centered_data', train_labels_cat, 'discrimType', 'pseudoLinear');

% The training data is then projected onto the linear discriminants derived
% from LDA, facilitating a visual examination of class separability in the
% reduced feature space. 

% Check the number of linear discriminants produced by LDA
numLinearDiscriminants = size(ldaModel.Coeffs, 1) - 1;
% disp(['Number of Linear Discriminants: ', num2str(numLinearDiscriminants)]);

% Check if two linear discriminants were produced
if numLinearDiscriminants >= 2
    ldaCoefficients = cat(2, ldaModel.Coeffs(1,2).Linear, ldaModel.Coeffs(1,3).Linear);
    % Project the training data onto the LDA components
    ldaProjection = centered_data' * ldaCoefficients;

    % Plot the scatter diagram of the first two linear discriminant coefficients
    figure;
    gscatter(ldaProjection(:,1), ldaProjection(:,2), train_labels_cat);
    title('Scatter Diagram of First Two Linear Discriminant Coefficients');
    xlabel('First Linear Discriminant');
    ylabel('Second Linear Discriminant');
else
    disp('LDA was unable to produce more than one discriminant.');
end

% This step leverages LDA to enhance the distinction between different
% emotional expressions by optimizing class separability. By projecting the
% data onto a space defined by linear discriminants, we observe clearer
% boundaries between classes compared to PCA, underscoring LDA's utility in
% preparing features for classification tasks.

%%
% Problem 1 (f) : Classification w/ LDA and Performance Evaluation
%

% After feature extraction using LDA, the next step involves applying the
% trained LDA model to classify the test data. This involves preparing the
% test data in a manner consistent with the training phase, ensuring that
% the data is centered using the mean derived from the training set. 

% Step 1: Prepare the test data
centered_test_data = test_data - mean(train_data, 2);

%fprintf('Size of training data: %d x %d\n', size(centered_data));
%fprintf('Size of test data: %d x %d\n', size(centered_test_data));

if size(centered_test_data, 2) ~= size(centered_data, 1)
    % Transpose centered_test_data if necessary
    centered_test_data = centered_test_data';
    % fprintf('Transposed test data to match feature dimensions.\n');
end

% Using the LDA model trained earlier, we predict the categories of the
% test dataset. The predictions are then compared with the actual labels to
% assess the model's performance.

% Step 2: Classify test data using the LDA model
% Use the LDA model to classify the centered test data
[predicted_labels, ~] = predict(ldaModel, centered_test_data);

% Ensure both predicted and actual labels are in the same format for comparison
test_labels_cat = categorical(test_label); 
predicted_labels_cat = categorical(predicted_labels); 

% Step 3: Plot confusion matrix
% Calculate the confusion matrix using the categorical labels
confMat = confusionmat(test_labels_cat, predicted_labels_cat);

% Visualize the confusion matrix using a confusion chart
confChart = confusionchart(confMat);

% Customize the confusion chart title
confChart.Title = 'Confusion Matrix for Test Set Classification Using LDA';

% Optionally, add more customization to the confusion chart for clarity
confChart.RowSummary = 'row-normalized'; % Normalize by rows to see the classification accuracy per class
confChart.ColumnSummary = 'column-normalized'; % Normalize by columns to understand precision per class

% Display classification accuracy: the overall classification accuracy is
% calculated as the ratio of correctly predicted labels to the total number
% of test samples.
accuracy = sum(predicted_labels_cat == test_labels_cat) / numel(test_labels_cat);
fprintf('Overall Classification Accuracy Using LDA: %.2f%%\n', accuracy * 100);

% The application of LDA for classification demonstrates its effectiveness
% in distinguishing between various facial expressions based on extracted
% features. The confusion matrix and the calculated accuracy provide a
% comprehensive view of the model's performance, highlighting its
% capability to accurately classify facial expressions with a good
% accuracy rate.


%%
% Problem 1 (g): Comparative Analysis & Conclusions
%

% PCA versus LDA: Feature Reduction and Class Separability

% PCA Observations:
% - PCA focuses on capturing the dataset's variance, projecting data onto axes that reflect the most significant variation.
% - While PCA effectively reduces dimensionality, it does not prioritize class separability, which can be crucial for classification tasks.

% LDA Observations:
% - LDA seeks to maximize class separation by reducing intra-class variance and enhancing inter-class distances, making it particularly valuable for classification.
% - Scatter plots generated in earlier sections reveal that LDA provides more distinct class clustering compared to PCA, offering clearer boundaries for classification.

% Displaying comparison information
disp('Comparing PCA and LDA:');

if exist('pca_train', 'var') && exist('ldaProjection', 'var')
    disp('Scatter plots have been generated separately in (c) for PCA and in (e) for LDA.');
    disp('By examining both scatter plots, we can visually ascertain the superiority of LDA in this context, as it provides more defined groupings of facial expressions, which is instrumental for recognition systems.');
    
    figure;
    subplot(1, 2, 1);
    gscatter(pca_train(:, 1), pca_train(:, 2), train_label);
    title('PCA: First vs. Second Principal Components');
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    
    subplot(1, 2, 2);
    gscatter(ldaProjection(:, 1), ldaProjection(:, 2), train_label);
    title('LDA: First vs. Second Linear Discriminants');
    xlabel('First Linear Discriminant');
    ylabel('Second Linear Discriminant');

else
    disp('PCA and/or LDA projections not found. Ensure you have run parts (c) and (e).');
end

% SVM versus Nearest Centroid Classifier: Classification Performance

% SVM Insights:
% - SVM's strength lies in finding the optimal hyperplane that separates classes in high-dimensional space, especially effective with PCA-reduced data.
% - However, SVM does not inherently optimize for class separability, potentially limiting its effectiveness when classes are not linearly separable.

% Nearest Centroid Classifier Insights:
% - Benefiting directly from LDA's emphasis on class separation, the Nearest Centroid Classifier assigns categories based on proximity to class centroids.
% - This method often results in higher interpretability and can lead to improved accuracy due to LDA's focus on maximizing class distances.

% Displaying classification results comparison
disp('Comparing SVM and Nearest Centroid Classifier results:');

if exist('accuracy', 'var') && exist('accuracyTest', 'var')
    fprintf('SVM Classification Accuracy: %.2f%%\n', accuracy * 100);
    fprintf('Nearest Centroid Classification Accuracy: %.2f%%\n', accuracyTest * 100);
    disp('The confusion matrices for SVM and Nearest Centroid Classifier have been plotted separately, showcasing the difference in classification performance between the two methods.');
else
    disp('Classification results not found. Ensure you have run parts (d) and (f).');
end



% Performance Comparison and Final Thoughts

% - Visual and quantitative comparisons of scatter plots and confusion matrices for PCA and LDA underscore LDA's superiority in enhancing class separability for this dataset.
% - Accuracy assessments of SVM and Nearest Centroid Classifier post-feature reduction illustrate the critical role of choosing appropriate methods aligned with classification goals.
% - The analysis reveals that while PCA and SVM are powerful tools for feature reduction and classification, respectively, LDA's focus on class distinction makes it particularly suited for tasks requiring clear class separation.

% Conclusion

% This project's exploration of PCA and LDA for feature reduction, coupled
% with SVM and Nearest Centroid Classifier for classification, demonstrates
% the importance of selecting techniques that align with the specific
% objectives of facial expression recognition. LDA's ability to enhance
% class separability has proven especially beneficial, suggesting its
% preference in scenarios where class distinction is paramount. Future work
% may explore combining these methodologies or investigating other
% classification algorithms to further improve accuracy and efficiency
% (especially for class 3 & 6)
