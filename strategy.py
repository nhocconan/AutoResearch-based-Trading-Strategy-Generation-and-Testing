#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Pullback_Entry_Volume
Hypothesis: Instead of chasing breakouts, enter on pullbacks to R1 (long) or S1 (short) after a strong directional move.
This reduces false breakouts and improves win rate. Uses 1d RSI regime filter: only long when RSI>50, short when RSI<50.
Volume confirmation on entry. Targets 20-30 trades/year to minimize fee drift. Works in both bull and bear by aligning with daily momentum.
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels R1 and S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily RSI for regime filter (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: pullback to R1 in bullish regime (RSI>50) with volume
            if close[i] <= r1_aligned[i] * 1.005 and rsi_aligned[i] > 50 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: pullback to S1 in bearish regime (RSI<50) with volume
            elif close[i] >= s1_aligned[i] * 0.995 and rsi_aligned[i] < 50 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches S1 or RSI turns bearish
            if close[i] <= s1_aligned[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches R1 or RSI turns bullish
            if close[i] >= r1_aligned[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Pullback_Entry_Volume"
timeframe = "4h"
leverage = 1.0