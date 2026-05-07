# 12h_Camarilla_R3S3_Volume_Chop
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and Choppiness filter.
# Long when price breaks above 12h Camarilla R3 level AND 1d volume spike AND Choppiness < 38.2 (trending).
# Short when price breaks below 12h S3 level AND 1d volume spike AND Choppiness < 38.2.
# Uses daily volume spike for momentum confirmation and Choppiness to avoid ranging markets.
# Designed for fewer trades (target: 15-25/year) to reduce fee drag and improve generalization.
# Works in both bull and bear markets by following 12h price action with volatility filter.
# 12h timeframe reduces trade frequency, Choppiness filter avoids false breakouts in ranges.
name = "12h_Camarilla_R3S3_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume spike and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_list = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[0] - low_1d[0]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_list.append(tr)
    atr_1d = np.array(atr_list)
    atr_ma_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_1d = 100 * np.log10(atr_ma_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_1d = np.where((highest_high_14 - lowest_low_14) > 0, chop_1d, 50.0)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Load 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 12h OHLC for Camarilla calculation
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    # Camarilla levels: R3/S3
    range_12h = prev_high_12h - prev_low_12h
    camarilla_r3 = prev_close_12h + range_12h * 1.1 / 4
    camarilla_s3 = prev_close_12h - range_12h * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above R3, volume spike, trending market (Chop < 38.2)
            long_condition = (close[i] > camarilla_r3_aligned[i]) and vol_spike_1d_aligned[i] and (chop_1d_aligned[i] < 38.2)
            # Short condition: break below S3, volume spike, trending market (Chop < 38.2)
            short_condition = (close[i] < camarilla_s3_aligned[i]) and vol_spike_1d_aligned[i] and (chop_1d_aligned[i] < 38.2)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or Choppiness turns to ranging (Chop > 61.8)
            if (close[i] < camarilla_s3_aligned[i]) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or Choppiness turns to ranging (Chop > 61.8)
            if (close[i] > camarilla_r3_aligned[i]) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals