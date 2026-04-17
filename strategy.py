#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R3/S3 breakout + volume confirmation + 1w EMA50 trend filter.
Long when price breaks above 1d Camarilla R3 with volume confirmation and price > 1w EMA50 (uptrend).
Short when price breaks below 1d Camarilla S3 with volume confirmation and price < 1w EMA50 (downtrend).
Exit when price returns to the 1d Camarilla midpoint (H4/L4) or reverses with volume.
Designed for fewer, higher-quality breakouts (target: 50-150 trades over 4 years) using wider Camarilla levels
to reduce noise and avoid false breakouts. Uses 1w EMA50 for strong trend filter to avoid counter-trend trades.
"""

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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R3 = close + 1.25*(high-low)
    # S3 = close - 1.25*(high-low)
    # H4/L4 = R1/S1 = close ± 0.833*(high-low)
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.25 * range_1d
    s3_1d = close_1d - 1.25 * range_1d
    h4_1d = close_1d + 0.833 * range_1d  # R1
    l4_1d = close_1d - 0.833 * range_1d  # S1
    midpoint_1d = (h4_1d + l4_1d) / 2  # equals close_1d
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume 50-period average for confirmation
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    # Align 1w EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 50-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_50[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with volume and uptrend (price > 1w EMA50)
            if (close[i] > r3_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Camarilla S3 with volume and downtrend (price < 1w EMA50)
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S3 with volume (reversal)
            if (close[i] <= midpoint_1d_aligned[i] or 
                (close[i] < s3_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R3 with volume (reversal)
            if (close[i] >= midpoint_1d_aligned[i] or 
                (close[i] > r3_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_1dCamarilla_R3S3_Breakout_Volume_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0