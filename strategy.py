#!/usr/bin/env python3
"""
6h_1D_Camarilla_R3S3_Fade_With_Volume
Hypothesis: On 6h timeframe, fade at Camarilla R3/S3 levels from daily pivot with volume confirmation.
Long when price crosses below S3 with volume > 1.5x 20-period average and RSI < 40.
Short when price crosses above R3 with volume > 1.5x 20-period average and RSI > 60.
Exit on touch of opposite level (R3 for longs, S3 for shorts) or opposite Camarilla level (S4/R4).
This targets mean reversion in ranging markets while avoiding low-volume false signals.
Works in both bull/bear markets as it fades extremes rather than following trends.
Target: 20-40 trades/year with strict entry conditions.
"""

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
    
    # Calculate volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Get 1D data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # Where C = (H+L+C)/3 (typical price)
    
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    r4 = typical_price + (hl_range * 1.1 / 2)
    r3 = typical_price + (hl_range * 1.1 / 4)
    s3 = typical_price - (hl_range * 1.1 / 4)
    s4 = typical_price - (hl_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Calculate RSI(14) on 6h for entry filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume avg and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_avg[i]) or np.isnan(rsi[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price crosses below S3 with volume confirmation and RSI not overbought
            if (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and
                volume[i] > 1.5 * vol_avg[i] and rsi[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses above R3 with volume confirmation and RSI not oversold
            elif (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg[i] and rsi[i] > 60):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price touches R3 (take profit) or breaks S4 (stop)
            if close[i] >= r3_aligned[i] or close[i] <= s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches S3 (take profit) or breaks R4 (stop)
            if close[i] <= s3_aligned[i] or close[i] >= r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_Camarilla_R3S3_Fade_With_Volume"
timeframe = "6h"
leverage = 1.0