#!/usr/bin/env python3
"""
4h_KAMA_Adaptive_Trend_R1S1_Breakout_VolumeSpike
Hypothesis: Use KAMA (adaptive moving average) to filter trend direction on 4h timeframe. 
Enter long when price breaks above R1 with KAMA upward slope and volume spike.
Enter short when price breaks below S1 with KAMA downward slope and volume spike.
Exit when price returns to opposite S1/R1 level or KAMA slope reverses.
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
Designed for 20-30 trades/year to minimize fee drag while maintaining edge in both bull/bear markets.
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
    
    # Get 4h data for KAMA trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close']
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) with ER=10, FC=2, SC=30
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # SC = Smoothing Constant = [ER * (fastest - slowest) + slowest]^2
    # where fastest = 2/(2+1) = 0.6667, slowest = 2/(30+1) = 0.0645
    change = np.abs(np.diff(close_4h, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close_4h, n=1)), axis=1)  # sum of abs changes
    
    # Handle first 10 values
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    
    # Calculate ER with safe division
    er = np.divide(change_padded, volatility_padded, 
                   out=np.full_like(change_padded, np.nan), 
                   where=volatility_padded!=0)
    
    # Calculate SC
    fastest = 2 / (2 + 1)
    slowest = 2 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # Start with close at index 9
    for i in range(10, len(close_4h)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Get 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate Camarilla levels for each day
    camarilla_range = (high_1d - low_1d)
    r1_level = close_1d + (1.1 * camarilla_range) / 12
    s1_level = close_1d - (1.1 * camarilla_range) / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(kama_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        kama_val = kama_aligned[i]
        vol_spike = volume_spike[i]
        
        # Calculate KAMA slope for trend direction
        if i > start_idx and not np.isnan(kama_aligned[i-1]):
            kama_slope = kama_val - kama_aligned[i-1]
        else:
            kama_slope = 0
        
        if position == 0:
            # Long: break above R1 with KAMA trending up and volume spike
            if price > r1 and kama_slope > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with KAMA trending down and volume spike
            elif price < s1 and kama_slope < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or KAMA turns down
            if price < s1 or kama_slope < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or KAMA turns up
            if price > r1 or kama_slope > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Adaptive_Trend_R1S1_Breakout_VolumeSpike"
timeframe = "4h"
leverage = 1.0