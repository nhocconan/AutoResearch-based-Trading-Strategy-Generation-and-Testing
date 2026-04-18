#!/usr/bin/env python3
"""
1h_TSIFlow_Confluence
Hypothesis: Combines 4h trend (EMA21), 1d momentum (RSI14 pullback), and 1h volume breakout.
Uses 4h for trend direction, 1d for momentum filter, and 1h for precise entry timing.
Designed to work in both bull and bear markets by requiring alignment across timeframes.
Target: 15-35 trades/year (~60-140 total over 4 years).
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
    
    # Get 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema21_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 21:
        ema21_4h[20] = np.mean(close_4h[0:21])
        alpha = 2 / (21 + 1)
        for i in range(21, len(close_4h)):
            ema21_4h[i] = close_4h[i] * alpha + ema21_4h[i-1] * (1 - alpha)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for momentum filter (RSI14)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    rsi14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        # Calculate RSI manually to avoid pandas dependency
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        
        # First average (simple mean)
        if len(gain) >= 14:
            avg_gain[13] = np.mean(gain[0:14])
            avg_loss[13] = np.mean(loss[0:14])
            
            # Wilder smoothing
            for i in range(14, len(close_1d)):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # 1h volume spike (volume > 1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # 1h price breakout above/below recent high/low (10-period)
    high_10 = np.full(n, np.nan)
    low_10 = np.full(n, np.nan)
    for i in range(10, n):
        high_10[i] = np.max(high[i-10:i])
        low_10[i] = np.min(low[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 10)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_10[i]) or np.isnan(low_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend, 1d momentum not overbought, 1h volume breakout above recent high
            if (close[i] > ema21_4h_aligned[i] and 
                rsi14_1d_aligned[i] < 70 and 
                vol_spike[i] and 
                close[i] > high_10[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, 1d momentum not oversold, 1h volume breakout below recent low
            elif (close[i] < ema21_4h_aligned[i] and 
                  rsi14_1d_aligned[i] > 30 and 
                  vol_spike[i] and 
                  close[i] < low_10[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend turns down or 1h price breaks below recent low
            if (close[i] < ema21_4h_aligned[i] or 
                close[i] < low_10[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend turns up or 1h price breaks above recent high
            if (close[i] > ema21_4h_aligned[i] or 
                close[i] > high_10[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_TSIFlow_Confluence"
timeframe = "1h"
leverage = 1.0