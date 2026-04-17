#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Camarilla R3/S3 breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 1w Camarilla R3 with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 1w Camarilla S3 with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 1w Camarilla midpoint (H4/L4) or reverses with volume.
Uses 1w for structure (major support/resistance), 1d for trend filter, and 12h for execution.
Designed to capture major breakouts with institutional volume while avoiding false breakouts in choppy markets.
Camarilla R3/S3 are stronger levels than R1/S1, reducing trade frequency and increasing reliability.
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
    
    # Get 1w data for Camarilla calculation (stronger weekly levels)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (based on prior 1w bar)
    # R3 = close + 1.25*(high-low)
    # S3 = close - 1.25*(high-low)
    # H4/L4 = (R1 + S1) / 2 = close (midpoint)
    range_1w = high_1w - low_1w
    r3_1w = close_1w + 1.25 * range_1w
    s3_1w = close_1w - 1.25 * range_1w
    midpoint_1w = close_1w  # Camarilla midpoint is close
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    midpoint_1w_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or 
            np.isnan(midpoint_1w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R3 with volume and uptrend (price > EMA50)
            if (close[i] > r3_1w_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S3 with volume and downtrend (price < EMA50)
            elif (close[i] < s3_1w_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S3 with volume (reversal)
            if (close[i] <= midpoint_1w_aligned[i] or 
                (close[i] < s3_1w_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R3 with volume (reversal)
            if (close[i] >= midpoint_1w_aligned[i] or 
                (close[i] > r3_1w_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wCamarilla_R3S3_Breakout_Volume_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0