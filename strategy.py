#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Resistance_Support_Breakout_Volume
Hypothesis: Breakouts above weekly pivot resistance or below weekly pivot support with volume confirmation. Works in bull markets via resistance breakouts, in bear markets via support breakdowns. Uses 1d timeframe with 1h trend filter to avoid counter-trend trades. Target: 10-25 trades/year on 1d timeframe with disciplined entry conditions.
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
    
    # 14-period ATR for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # 1h EMA50 trend filter (from higher timeframe)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema50_1h = np.full(len(close_1h), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1h)):
        if i == 50:
            ema50_1h[i] = np.mean(close_1h[0:51])
        else:
            ema50_1h[i] = close_1h[i] * k + ema50_1h[i-1] * (1 - k)
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and 1h uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema50_1h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and 1h downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema50_1h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly pivot or 1h trend turns down
            if (close[i] < pivot_aligned[i] or close[i] < ema50_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly pivot or 1h trend turns up
            if (close[i] > pivot_aligned[i] or close[i] > ema50_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_Resistance_Support_Breakout_Volume"
timeframe = "1d"
leverage = 1.0