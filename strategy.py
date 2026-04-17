#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d/1w Camarilla R3/S3 breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 1d Camarilla R3 with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 1d Camarilla S3 with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 1d Camarilla midpoint (H4/L4) or reverses with volume.
Uses 1d timeframe for structure and trend filter, 6h for entry timing and volume confirmation.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Camarilla levels provide precise support/resistance based on prior day's range, effective in both trending and ranging markets.
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
    
    # Get 1d data for Camarilla calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.25*(high-low)
    # R2 = close + 1.166*(high-low)
    # R1 = close + 0.833*(high-low)
    # S1 = close - 0.833*(high-low)
    # S2 = close - 1.166*(high-low)
    # S3 = close - 1.25*(high-low)
    # S4 = close - 1.5*(high-low)
    # Midpoint H4/L4 = (R1 + S1) / 2 = close
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.25 * range_1d
    s3_1d = close_1d - 1.25 * range_1d
    midpoint_1d = close_1d  # Camarilla midpoint is close
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with volume and uptrend (price > EMA50)
            if (close[i] > r3_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with volume and downtrend (price < EMA50)
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
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

name = "6h_1dCamarilla_R3S3_Breakout_Volume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0