
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


def undersampling(X_train, y_train):
    print("undersampling: ")
    rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def oversampling(X_train, y_train):
    print("oversampling: ")
    # smote = SMOTE()
    # smote = SMOTE(k_neighbors=2, random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    oversample = RandomOverSampler()
    X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)

    return X_resampled, y_resampled