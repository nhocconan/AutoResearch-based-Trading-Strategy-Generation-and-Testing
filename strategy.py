#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d Supertrend(10,3) trend filter and volume confirmation (>1.8x avg volume). 
# Uses ATR(20) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Supertrend on 1d provides robust trend detection that works in both bull and bear markets.
# Donchian breakouts capture institutional breakout/breakdown points with proven efficacy on BTC/ETH.
# Volume confirmation filters false breakouts. ATR trailing stop manages risk without look-ahead.

name = "4h_Donchian20_1dSupertrend_Trend_VolumeSpike_ATRStop_v1"
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
    
    # Calculate ATR(20) for trailing stop and Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend(10,3) on 1d
    # ATR(10)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Upper and Lower Bands
    hl2_1d = (high_1d + low_1d) / 2
    upper_band_1d = hl2_1d + (3.0 * atr_1d)
    lower_band_1d = hl2_1d - (3.0 * atr_1d)
    
    # Initialize Supertrend
    supertrend_1d = np.full(len(close_1d), np.nan)
    direction_1d = np.full(len(close_1d), np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Start calculation after sufficient ATR data
    for i in range(10, len(close_1d)):
        # Upper Band
        if i == 10:
            upper_band_1d[i] = hl2_1d[i] + (3.0 * atr_1d[i])
            lower_band_1d[i] = hl2_1d[i] - (3.0 * atr_1d[i])
        else:
            upper_band_1d[i] = hl2_1d[i] + (3.0 * atr_1d[i])
            lower_band_1d[i] = hl2_1d[i] - (3.0 * atr_1d[i])
            
            # Adjust bands based on previous close
            if close_1d[i-1] <= upper_band_1d[i-1]:
                upper_band_1d[i] = min(upper_band_1d[i], upper_band_1d[i-1])
            if close_1d[i-1] >= lower_band_1d[i-1]:
                lower_band_1d[i] = max(lower_band_1d[i], lower_band_1d[i-1])
        
        # Determine trend direction
        if close_1d[i] > upper_band_1d[i-1]:
            direction_1d[i] = 1
        elif close_1d[i] < lower_band_1d[i-1]:
            direction_1d[i] = -1
        else:
            direction_1d[i] = direction_1d[i-1]
        
        # Set Supertrend value
        if direction_1d[i] == 1:
            supertrend_1d[i] = lower_band_1d[i]
        else:
            supertrend_1d[i] = upper_band_1d[i]
    
    # Align Supertrend and direction to 4h timeframe (wait for daily bar to close)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND 1d Supertrend uptrend AND volume > 1.8x average
            if (close[i] > highest_high[i] and 
                direction_1d_aligned[i] == 1 and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Donchian lower band AND 1d Supertrend downtrend AND volume > 1.8x average
            elif (close[i] < lowest_low[i] and 
                  direction_1d_aligned[i] == -1 and 
                  volume[i] > 1.8 * avg_volume[i]):
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
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
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
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
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