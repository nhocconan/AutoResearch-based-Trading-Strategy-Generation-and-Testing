#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-day EMA trend filter and volume confirmation
# Long when price > 1-day EMA100 AND Choppiness Index(14) < 38.2 (trending) AND volume > 1.5x 20-period average
# Short when price < 1-day EMA100 AND Choppiness Index(14) < 38.2 (trending) AND volume > 1.5x 20-period average
# Exit when Choppiness Index(14) > 61.8 (range) OR price crosses EMA100
# This avoids range-bound losses and captures trending moves with volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index on 12h (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * 14))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0)  # Default to neutral
    mask = (range14 > 0) & (~np.isnan(range14))
    chop[mask] = 100 * np.log10(sum_atr14[mask] / (np.log10(range14[mask]) * 14))
    
    # Calculate daily EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (100 for EMA100 + buffer)
    start = 110
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price above EMA100 AND trending market (CHOP < 38.2) AND volume confirmation
            if (price > ema100_1d_aligned[i] and chop[i] < 38.2 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below EMA100 AND trending market (CHOP < 38.2) AND volume confirmation
            elif (price < ema100_1d_aligned[i] and chop[i] < 38.2 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: range market (CHOP > 61.8) OR price crosses below EMA100
            if chop[i] > 61.8 or price < ema100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: range market (CHOP > 61.8) OR price crosses above EMA100
            if chop[i] > 61.8 or price > ema100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Chop_EMA100_Volume"
timeframe = "12h"
leverage = 1.0