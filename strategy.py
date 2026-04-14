#!/usr/bin/env python3
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
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly volume moving average (15-period) for volume filter
    volume_1w_series = pd.Series(volume_1w)
    volume_ma_1w = volume_1w_series.rolling(window=15, min_periods=15).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.28
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]):
            continue
        
        # Skip low volume periods (volume < 60% of 15-period MA)
        if volume[i] < 0.6 * volume_ma_1w_aligned[i]:
            continue
        
        # Get previous week's data (1w index)
        if i >= 1:
            prev_close = close_1w[i-1]
            prev_high = high_1w[i-1]
            prev_low = low_1w[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align S3/R3 to weekly timeframe (constant values for the week)
            s3_array = np.full(len(df_1w), s3)
            r3_array = np.full(len(df_1w), r3)
            s3_1w = align_htf_to_ltf(prices, df_1w, s3_array)[i]
            r3_1w = align_htf_to_ltf(prices, df_1w, r3_array)[i]
            
            if position == 0:
                # Long: Price closes above S3 with volume, above EMA20
                if close[i] > s3_1w and volume[i] > volume_ma_1w_aligned[i] and close[i] > ema20_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price closes below R3 with volume, below EMA20
                elif close[i] < r3_1w and volume[i] > volume_ma_1w_aligned[i] and close[i] < ema20_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price closes below S3 or trend changes (price below EMA20)
                if close[i] < s3_1w or close[i] < ema20_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price closes above R3 or trend changes (price above EMA20)
                if close[i] > r3_1w or close[i] > ema20_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "1d_1w_Pivot_S3R3_Rejection_Volume_EMA20_Filter_v1"
timeframe = "1d"
leverage = 1.0