#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla R3/S3 breakout + volume confirmation + 1d EMA34 trend filter.
Long when price breaks above 1d Camarilla R3 with volume confirmation and price > 1d EMA34 (strong uptrend).
Short when price breaks below 1d Camarilla S3 with volume confirmation and price < 1d EMA34 (strong downtrend).
Exit when price returns to the 1d Camarilla midpoint (H4/L4) or reverses with volume.
Uses 1d timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Camarilla R3/S3 levels represent stronger support/resistance than R1/S1, leading to fewer but higher-quality trades.
"""

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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.166 * range_1d  # R3 level
    s3_1d = close_1d - 1.166 * range_1d  # S3 level
    midpoint_1d = close_1d  # Camarilla midpoint is close
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with volume and strong uptrend (price > EMA34)
            if (close[i] > r3_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with volume and strong downtrend (price < EMA34)
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S3 with volume (reversal)
            if (close[i] <= midpoint_1d_aligned[i] or 
                (close[i] < s3_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R3 with volume (reversal)
            if (close[i] >= midpoint_1d_aligned[i] or 
                (close[i] > r3_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R3S3_Breakout_Volume_EMA34_Trend"
timeframe = "4h"
leverage = 1.0