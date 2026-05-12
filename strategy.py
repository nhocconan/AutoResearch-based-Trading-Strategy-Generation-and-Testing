# 4h_Camarilla_Pivot_R3S3_Breakout_12hEMA50_Trend
# Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# R3/S3 are strong pivot levels from the previous day; breakout indicates momentum.
# 12h EMA50 ensures trades align with higher timeframe trend, reducing whipsaw.
# Volume filter confirms institutional participation. Designed for low frequency (20-50 trades/year)
# to avoid fee drag. Works in both bull and bear markets by following higher timeframe trend.

name = "4h_Camarilla_Pivot_R3S3_Breakout_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the previous period.
    R4 = C + ((H-L) * 1.5000)
    R3 = C + ((H-L) * 1.2500)
    R2 = C + ((H-L) * 1.1666)
    R1 = C + ((H-L) * 1.0833)
    PP = (H + L + C) / 3
    S1 = C - ((H-L) * 1.0833)
    S2 = C - ((H-L) * 1.1666)
    S3 = C - ((H-L) * 1.2500)
    S4 = C - ((H-L) * 1.5000)
    """
    typical = (high + low + close) / 3
    range_ = high - low
    r4 = close + range_ * 1.5000
    r3 = close + range_ * 1.2500
    r2 = close + range_ * 1.1666
    r1 = close + range_ * 1.0833
    pp = typical
    s1 = close - range_ * 1.0833
    s2 = close - range_ * 1.1666
    s3 = close - range_ * 1.2500
    s4 = close - range_ * 1.5000
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
        high_1d[:-1], low_1d[:-1], close_1d[:-1]
    )
    
    # For current day, we use the previous day's levels
    # Pad with NaN for the first element (no previous day)
    r1 = np.concatenate([np.array([np.nan]), r1])
    r2 = np.concatenate([np.array([np.nan]), r2])
    r3 = np.concatenate([np.array([np.nan]), r3])
    s1 = np.concatenate([np.array([np.nan]), s1])
    s2 = np.concatenate([np.array([np.nan]), s2])
    s3 = np.concatenate([np.array([np.nan]), s3])
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA and indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above R3 AND above 12h EMA50 AND volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND below 12h EMA50 AND volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema_50_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below S3 OR below 12h EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3 OR above 12h EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals