#!/usr/bin/env python3
# 1d_camarilla_pivot_breakout_volume_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels with volume confirmation.
# Long when price breaks above weekly R4 with volume > 1.5x 20-period average.
# Short when price breaks below weekly S4 with volume > 1.5x 20-period average.
# Exit when price closes back inside weekly R3/S3 levels.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong breakouts in both bull and bear markets while avoiding false signals.
# Target: 10-25 trades/year (40-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_breakout_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    pivot_1w = typical_price_1w
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly levels to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price closes back below weekly R3 (take profit) or below weekly S4 (stop)
            if close[i] < r3_1w_aligned[i] or close[i] < s4_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above weekly S3 (take profit) or above weekly R4 (stop)
            if close[i] > s3_1w_aligned[i] or close[i] > r4_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation (weekly only)
            bullish_breakout = (close[i] > r4_1w_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < s4_1w_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals