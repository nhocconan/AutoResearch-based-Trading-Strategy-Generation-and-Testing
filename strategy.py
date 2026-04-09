#!/usr/bin/env python3
# 4h_camarilla_pullback_volume_v1
# Hypothesis: 4h strategy buying pullbacks to daily Camarilla support/resistance with volume confirmation.
# Long when price retraces to daily S3 with volume > 1.3x average and closes above it.
# Short when price retraces to daily R3 with volume > 1.3x average and closes below it.
# Exit when price reaches opposite Camarilla level (S4/R4) or shows reversal weakness.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong intraday moves within the daily range while avoiding breakout failures.
# Target: 30-60 trades/year (120-240 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pullback_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    pivot_1d = typical_price_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align all levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price reaches daily R4 (target) or shows weakness below S3
            if close[i] >= r4_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches daily S4 (target) or shows weakness above R3
            if close[i] <= s4_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for pullback to Camarilla levels with volume confirmation
            bullish_pullback = (low[i] <= s3_1d_aligned[i] and close[i] > s3_1d_aligned[i]) and volume_confirmed
            bearish_pullback = (high[i] >= r3_1d_aligned[i] and close[i] < r3_1d_aligned[i]) and volume_confirmed
            
            if bullish_pullback:
                position = 1
                signals[i] = 0.25
            elif bearish_pullback:
                position = -1
                signals[i] = -0.25
    
    return signals