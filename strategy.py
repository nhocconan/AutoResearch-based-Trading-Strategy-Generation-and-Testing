#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume
Uses Camarilla pivot levels from 1d to identify key support/resistance zones.
Breaks above/below key levels with volume confirmation to capture breakouts.
Uses RSI(20) filter to avoid counter-trend entries in strong trends.
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining pivot structure with momentum.
"""

name = "12h_1d_camarilla_breakout_volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels calculation
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + (range_1d * 1.1 / 2)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Use R3 and S3 as key breakout levels
    r3_level = r3
    s3_level = s3
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume confirmation on 12h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # RSI filter to avoid counter-trend entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=20, min_periods=20).mean()
    avg_loss = loss.rolling(window=20, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R3 with volume and RSI not overbought
        if (close[i] > r3_aligned[i] and vol_confirm[i] and 
            rsi_values[i] < 70 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S3 with volume and RSI not oversold
        elif (close[i] < s3_aligned[i] and vol_confirm[i] and 
              rsi_values[i] > 30 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: return to pivot or opposite level
        elif position == 1 and (close[i] <= pivot[i] or close[i] < s3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pivot[i] or close[i] > r3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals