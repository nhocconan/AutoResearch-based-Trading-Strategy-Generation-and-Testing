#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Choppiness Index regime filter and Donchian breakout.
# Uses 1d Choppiness Index to filter ranging markets (CHOP > 61.8) for mean-reversion at Donchian bands.
# In trending markets (CHOP < 38.2), follows Donchian breakout.
# Volume confirmation ensures breakout strength. Designed for low trade frequency (<30/year).
name = "12h_1d_Chop_Donchian_MeanRev_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index and Donchian calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14)
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        chop_val = chop_aligned[i]
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        
        if position == 0:
            # In ranging market (CHOP > 61.8): mean reversion at Donchian bands
            if chop_val > 61.8:
                if close[i] <= dl and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= dh and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # In trending market (CHOP < 38.2): follow Donchian breakout
            elif chop_val < 38.2:
                if close[i] > dh and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < dl and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # In transition zone (38.2 <= CHOP <= 61.8): no new entries
            
        elif position == 1:
            # Long position exit conditions
            if chop_val > 61.8 and close[i] >= dh:  # Take profit at upper band in ranging
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] < dl:  # Stop loss below lower band in trending
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position exit conditions
            if chop_val > 61.8 and close[i] <= dl:  # Take profit at lower band in ranging
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] > dh:  # Stop loss above upper band in trending
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals