# 4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Strategy: Use 1-day Camarilla pivot levels (S3/R3) for breakout entries on 4h timeframe
# Long when price breaks above R3 with volume spike and 1d trend filter (price > EMA200)
# Short when price breaks below S3 with volume spike and 1d trend filter (price < EMA200)
# Exit when price reverses to opposite S3/R3 level or trend fails
# Uses volatility-based volume confirmation and trend alignment to reduce false breakouts
# Designed for 4h timeframe with institutional level breakouts and volume confirmation

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels (S3, R3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges for previous day
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_ = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels: S3 = close - 1.1*(high-low)/6, R3 = close + 1.1*(high-low)/6
    s3 = close_1d[:-1] - 1.1 * range_ / 6
    r3 = close_1d[:-1] + 1.1 * range_ / 6
    
    # Align S3 and R3 to 4h timeframe (previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], r3)
    
    # Calculate 1-day EMA200 for trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate volume spike detector (volume > 2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike and uptrend
            if (close[i] > r3_aligned[i] and volume_spike[i] and 
                close[i] > ema_200_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike and downtrend
            elif (close[i] < s3_aligned[i] and volume_spike[i] and 
                  close[i] < ema_200_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend fails
            if (close[i] < s3_aligned[i] or close[i] < ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend fails
            if (close[i] > r3_aligned[i] or close[i] > ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals