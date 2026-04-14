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
    
    # Load daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (more precise for intraday)
    # Camarilla formula: based on previous day's range
    range_1d = high_1d - low_1d
    close_prev = close_1d  # Using same day close for calculation (will be aligned properly)
    
    # Camarilla levels
    # Resistance levels
    r1 = close_prev + (range_1d * 1.1 / 12)
    r2 = close_prev + (range_1d * 1.1 / 6)
    r3 = close_prev + (range_1d * 1.1 / 4)
    r4 = close_prev + (range_1d * 1.1 / 2)
    # Support levels
    s1 = close_prev - (range_1d * 1.1 / 12)
    s2 = close_prev - (range_1d * 1.1 / 6)
    s3 = close_prev - (range_1d * 1.1 / 4)
    s4 = close_prev - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 10-period daily ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 10:
        atr_1d[9] = np.mean(tr[:10])
        for i in range(10, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 9 + tr[i]) / 10
    
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (15-period average on 4h)
    vol_ma_15 = np.full_like(volume, np.nan)
    if len(volume) >= 15:
        for i in range(14, len(volume)):
            vol_ma_15[i] = np.mean(volume[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to manage drawdown
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or
            np.isnan(s1_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(atr_4h[i]) or np.isnan(vol_ma_15[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.6% of price)
        if atr_4h[i] < 0.006 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 15-period average
        if vol_ma_15[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_15[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 1.8
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation
            if (close[i] > r3_4h[i] and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S3 with volume confirmation
            elif (close[i] < s3_4h[i] and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S2 (tighter stop)
            if close[i] < s2_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R2 (tighter stop)
            if close[i] > r2_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_R3S3_Breakout_Volume"
timeframe = "4h"
leverage = 1.0