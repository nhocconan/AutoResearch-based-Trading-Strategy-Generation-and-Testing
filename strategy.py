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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA20 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate daily volume moving average (15-period) for volume filter
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=15, min_periods=15).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            continue
        
        # Skip low volume periods (volume < 60% of 15-period MA)
        if volume[i] < 0.6 * volume_ma_1d_aligned[i]:
            continue
        
        # Skip high volatility periods (ATR > 2 * 20-period ATR average)
        if i >= 20:
            atr_avg = np.mean(atr_1d_aligned[i-20:i])
            if atr_1d_aligned[i] > 2.0 * atr_avg:
                continue
        
        # Get previous day's data (1d index)
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align S3/R3 to daily timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_1d = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_1d = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            if position == 0:
                # Long: Price closes above S3 with volume, above EMA20, not too volatile
                if close[i] > s3_1d and volume[i] > volume_ma_1d_aligned[i] and close[i] > ema20_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price closes below R3 with volume, below EMA20, not too volatile
                elif close[i] < r3_1d and volume[i] > volume_ma_1d_aligned[i] and close[i] < ema20_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price closes below S3 or trend changes (price below EMA20) or volatility spike
                if close[i] < s3_1d or close[i] < ema20_1d_aligned[i] or (i >= 20 and atr_1d_aligned[i] > 2.0 * np.mean(atr_1d_aligned[i-20:i])):
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price closes above R3 or trend changes (price above EMA20) or volatility spike
                if close[i] > r3_1d or close[i] > ema20_1d_aligned[i] or (i >= 20 and atr_1d_aligned[i] > 2.0 * np.mean(atr_1d_aligned[i-20:i])):
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "1d_Pivot_S3R3_Rejection_Volume_EMA20_ATR_Filter_v1"
timeframe = "1d"
leverage = 1.0