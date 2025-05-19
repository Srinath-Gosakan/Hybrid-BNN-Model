import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_files(logon, email, file, device, http, psych):
    # --- LOGON ---
    logon['date'] = pd.to_datetime(logon['date'])
    logon['day'] = logon['date'].dt.day
    logon['hour'] = logon['date'].dt.hour
    logon_features = logon.groupby('user').agg(
        L1=('pc', 'nunique'),
        L2=('activity', lambda x: (x == 'Logon').sum()),
        L3=('activity', lambda x: (x == 'Logoff').sum()),
        L4=('hour', lambda x: x[logon.loc[x.index, 'activity'] == 'Logon'].mode()[0] if not x[logon.loc[x.index, 'activity'] == 'Logon'].empty else -1),
        L5=('hour', lambda x: x[logon.loc[x.index, 'activity'] == 'Logoff'].mode()[0] if not x[logon.loc[x.index, 'activity'] == 'Logoff'].empty else -1),
        L6=('day', lambda x: (x >= 5).sum()),
        L7=('day', lambda x: (x < 5).sum()),
        L8=('pc', lambda x: x.value_counts().idxmax()),
        L9=('hour', lambda x: ((x < 6) | (x > 18)).sum()),
        L10=('hour', lambda x: ((x >= 6) & (x <= 18)).sum()),
    ).reset_index()

    # --- EMAIL ---
    email['date'] = pd.to_datetime(email['date'])
    email['day'] = email['date'].dt.day
    email['hour'] = email['date'].dt.hour
    email['attachments'] = pd.to_numeric(email['attachments'], errors='coerce').fillna(0).astype(int)
    email_features = email.groupby('user').agg(
        E1=('pc', 'nunique'),
        E2=('cc', 'count'),
        E3=('bcc', 'count'),
        E4=('size', 'mean'),
        E5=('attachments', 'sum'),
        E8=('hour', lambda x: ((x < 6) | (x > 18)).sum()),
        E9=('pc', 'nunique'),
        E10=('to', lambda x: x.value_counts().idxmax() if not x.empty else 'Unknown'),
        E12=('to', lambda x: x.dropna().astype(str).str.contains('gmail.com|yahoo.com|msn.com|juno.com|.net').sum()),
    ).reset_index()

    # --- HTTP ---
    http['date'] = pd.to_datetime(http['date'])
    http['day'] = http['date'].dt.day
    http['hour'] = http['date'].dt.hour
    http_features = http.groupby('user').agg(
        H1=('url', 'nunique'),
        H2=('activity', 'count'),
        H3=('hour', lambda x: ((x < 6) | (x > 18)).sum()),
        H4=('day', lambda x: (x >= 5).sum()),
        H7=('url', lambda x: x.str.contains('examiner|discovery|foodnetwork|thechive|wsj', na=False).sum()),
        H8=('url', lambda x: x.str.contains('soundcloud|m-w|youtube|cafemom|netflix', na=False).sum()),
        H9=('url', lambda x: x.str.len().mean()),
    ).reset_index()

    # --- FILE ---
    file['date'] = pd.to_datetime(file['date'])
    file['day'] = file['date'].dt.day
    file['hour'] = file['date'].dt.hour
    file['to_removable_media'] = file['to_removable_media'].astype(bool)
    file['from_removable_media'] = file['from_removable_media'].astype(bool)
    file_features = file.groupby('user').agg(
        F1=('filename', 'nunique'),
        F4=('from_removable_media', 'sum'),
        F5=('hour', lambda x: ((x < 6) | (x > 18)).sum()),
        F8=('activity', lambda x: x.str.contains('copy', case=False, na=False).sum()),
        F10=('content', lambda x: x.dropna().astype(str).str.count('confidential|password|sensitive|top secret|private|internal use|not for public|restricted').sum()),
        F11=('content', lambda x: x.dropna().astype(str).str.len().mean()),
    ).reset_index()

    # --- DEVICE ---
    device['date'] = pd.to_datetime(device['date'])
    device['day'] = device['date'].dt.day
    device['hour'] = device['date'].dt.hour
    device_features = device.groupby('user').agg(
        D1=('pc', 'nunique'),
        D2=('activity', 'count'),
        D3=('hour', lambda x: ((x < 6) | (x > 18)).sum()),
        D4=('day', lambda x: (x >= 5).sum()),
        D5=('activity', lambda x: x.str.contains('connect', case=False, na=False).sum()),
        D6=('activity', lambda x: x.str.contains('disconnect', case=False, na=False).sum()),
    ).reset_index()

    # --- PSYCHOMETRIC ---
    psych_features = psych[['user_id', 'O', 'C', 'E', 'A', 'N']].copy()
    psych_features.rename(columns={'user_id': 'user'}, inplace=True)

    # --- COMBINE ---
    combined_df = logon_features
    combined_df = combined_df.merge(email_features, on='user', how='outer')
    combined_df = combined_df.merge(http_features, on='user', how='outer')
    combined_df = combined_df.merge(file_features, on='user', how='outer')
    combined_df = combined_df.merge(device_features, on='user', how='outer')
    combined_df = combined_df.merge(psych_features, on='user', how='outer')

    # --- ENCODING (skip 'user') ---
    le = LabelEncoder()
    for col in combined_df.select_dtypes(include='object').columns:
        if col != 'user':
            combined_df[col] = le.fit_transform(combined_df[col].astype(str))

    # --- FILL NA + SCALE ---
    combined_df.fillna(0, inplace=True)
    scaler = MinMaxScaler()
    features_only = combined_df.drop(columns=['user'])
    scaled_features = scaler.fit_transform(features_only)

    # --- FINAL DATAFRAME ---
    processed_df = pd.DataFrame(scaled_features, columns=features_only.columns)
    processed_df['user'] = combined_df['user'].values

    return processed_df