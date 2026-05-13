#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with volume spike and 1d chop regime filter.
# Long when price breaks above R3 AND volume > 2.0x average AND chop > 61.8 (range).
# Short when price breaks below S3 AND volume > 2.0x average AND chop > 61.8 (range).
# Uses ATR(14) trailing stop (1.5x) for risk control. Discrete sizing 0.25.
# Choppiness index > 61.8 identifies ranging markets where Camarilla levels work best.
# Volume spike confirms breakout validity. Target: 75-200 total trades over 4 years (19-50/year) on 4h.

name = "4h_Camarilla_R3S3_Breakout_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 4h data (14-period)
    def calculate_chop(high, low, close, window=14):
        atr_sum = pd.Series(np.maximum(high - low, 
                                      np.maximum(np.abs(high - np.roll(close, 1)),
                                                 np.abs(low - np.roll(close, 1))))).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        # Handle division by zero or invalid values
        chop = np.where((highest_high - lowest_low) == 0, 50, chop)
        return chop.values
    
    chop = calculate_chop(high, low, close, 14)
    
    # Get 1d data for Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    
    # Align 1d Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND volume > 2.0x average AND chop > 61.8 (range)
            if (close[i] > r3_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i] and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below S3 AND volume > 2.0x average AND chop > 61.8 (range)
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i] and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 1.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 1.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals