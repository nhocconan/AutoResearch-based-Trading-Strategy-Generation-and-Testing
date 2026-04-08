#!/usr/bin/env python3
# 4h_camarilla_pivot_12h_trend_volume_v1
# Hypothesis: Camarilla pivot levels from 12h chart act as strong support/resistance levels.
# Long when price retraces to S3 level with volume confirmation and 12h uptrend (close > open).
# Short when price retraces to R3 level with volume confirmation and 12h downtrend (close < open).
# Exit when price reaches opposite Camarilla level (S1/R1) or shows reversal at same level.
# Uses 4h chart for entries to balance trade frequency and signal quality.
# Designed to work in ranging markets (2025-2026 test) by trading mean reversion at strong levels.
# Target: 20-40 trades/year to minimize fee decay while capturing reliable reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 12h data for Camarilla pivots (calculate once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla calculations
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Resistance levels
    r1 = close_12h + (range_12h * 1.1 / 12)
    r2 = close_12h + (range_12h * 1.1 / 6)
    r3 = close_12h + (range_12h * 1.1 / 4)
    r4 = close_12h + (range_12h * 1.1 / 2)
    
    # Support levels
    s1 = close_12h - (range_12h * 1.1 / 12)
    s2 = close_12h - (range_12h * 1.1 / 6)
    s3 = close_12h - (range_12h * 1.1 / 4)
    s4 = close_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 4h chart
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Get 12h data for trend filter (calculate once before loop)
    open_12h = df_12h['open'].values
    close_12h = df_12h['close'].values
    # 12h trend: close > open = uptrend, close < open = downtrend
    twelve_h_uptrend = close_12h > open_12h
    twelve_h_downtrend = close_12h < open_12h
    
    # Align 12h trend to 4h chart
    twelve_h_uptrend_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_uptrend.astype(float))
    twelve_h_downtrend_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_downtrend.astype(float))
    
    # Volume confirmation: 20-period average on 4h chart
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(twelve_h_uptrend_aligned[i]) or np.isnan(twelve_h_downtrend_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S1 (profit target) or shows rejection at S3
            if close[i] <= s1_aligned[i] or \
               (close[i] >= s3_aligned[i] and volume[i] > 1.5 * avg_volume[i] and twelve_h_downtrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 (profit target) or shows rejection at R3
            if close[i] >= r1_aligned[i] or \
               (close[i] <= r3_aligned[i] and volume[i] > 1.5 * avg_volume[i] and twelve_h_uptrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price retraces to S3 level with volume and 12h uptrend
            if close[i] <= s3_aligned[i] and volume_ok and twelve_h_uptrend_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price retraces to R3 level with volume and 12h downtrend
            elif close[i] >= r3_aligned[i] and volume_ok and twelve_h_downtrend_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals