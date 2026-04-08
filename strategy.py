#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_volume_v1
# Hypothesis: Use daily Camarilla pivot levels with volume confirmation on 4h timeframe.
# Long when price touches or crosses above S3 level with volume > 2x average and bullish bias.
# Short when price touches or crosses below R3 level with volume > 2x average and bearish bias.
# Exit on opposite signal or when price crosses H4/L4 levels.
# Uses mean-reversion at extreme pivot levels with volume filter to avoid false signals.
# Target: 20-30 trades/year to minimize fee drag while capturing reversals at key levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Calculate daily Camarilla pivot levels
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    S1 = close_1d - (range_1d * 1.1 / 12)
    S2 = close_1d - (range_1d * 1.1 / 6)
    S3 = close_1d - (range_1d * 1.1 / 4)
    S4 = close_1d - (range_1d * 1.1 / 2)
    
    R1 = close_1d + (range_1d * 1.1 / 12)
    R2 = close_1d + (range_1d * 1.1 / 6)
    R3 = close_1d + (range_1d * 1.1 / 4)
    R4 = close_1d + (range_1d * 1.1 / 2)
    
    # Align daily levels to 4h timeframe (1d -> 4h)
    S3_1d = S3
    R3_1d = R3
    S4_1d = S4
    R4_1d = R4
    
    S3_4h = align_htf_to_ltf(prices, df_1d, S3_1d)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3_1d)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4_1d)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4_1d)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Simple trend filter: price vs 50-period EMA on 4h
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend = close > ema50
    downtrend = close < ema50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(S3_4h[i]) or np.isnan(R3_4h[i]) or \
           np.isnan(S4_4h[i]) or np.isnan(R4_4h[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(ema50[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S4 or opposite signal
            if close[i] < S4_4h[i] or \
               (close[i] < R3_4h[i] and volume[i] > 2.0 * avg_volume[i] and downtrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R4 or opposite signal
            if close[i] > R4_4h[i] or \
               (close[i] > S3_4h[i] and volume[i] > 2.0 * avg_volume[i] and uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 2x average volume
            volume_ok = volume[i] > 2.0 * avg_volume[i]
            
            # Long entry: price touches or crosses above S3 with volume and uptrend bias
            if close[i] >= S3_4h[i] and volume_ok and uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or crosses below R3 with volume and downtrend bias
            elif close[i] <= R3_4h[i] and volume_ok and downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals