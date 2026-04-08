#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_reversal_v2
# Hypothesis: Camarilla pivot levels on 12h derived from 1d OHLC, with reversal entries when price touches S3/R3 levels with volume confirmation and RSI filter.
# Long when price touches S3 (1.1*close - 0.1*high) with RSI<30 and volume>1.5x average.
# Short when price touches R3 (1.1*high - 0.1*close) with RSI>70 and volume>1.5x average.
# Exit when price reaches S2/R2 or opposite S3/R3 level.
# Designed to capture mean reversions at extreme intraday levels in both bull and bear markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_reversal_v2"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 for entries, R2 and S2 for exits
    high_low = high_1d - low_1d
    r3 = close_1d + 1.1 * high_low
    s3 = close_1d - 1.1 * high_low
    r2 = close_1d + 0.5 * high_low
    s2 = close_1d - 0.5 * high_low
    
    # Align daily levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate RSI(14) on 12h for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or 
            np.isnan(rsi[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S2 or touches R3 (opposite extreme)
            if close[i] <= s2_12h[i] or close[i] >= r3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R2 or touches S3 (opposite extreme)
            if close[i] >= r2_12h[i] or close[i] <= s3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Reversal entries: price touches S3 (long) or R3 (short) with RSI confirmation
            long_condition = (close[i] <= s3_12h[i]) and (rsi[i] < 30) and volume_ok
            short_condition = (close[i] >= r3_12h[i]) and (rsi[i] > 70) and volume_ok
            
            if long_condition:
                position = 1
                signals[i] = 0.25
            elif short_condition:
                position = -1
                signals[i] = -0.25
    
    return signals