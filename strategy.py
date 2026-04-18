#!/usr/bin/env python3
"""
1d_RSI_Extremes_WeeklyTrend_Filtered
Hypothesis: RSI extremes (<30 for long, >70 for short) combined with weekly trend filter (price above/below weekly EMA50) 
provides high-probability mean-reversion entries in ranging markets while avoiding counter-trend trades in strong trends.
Volume confirmation filters out low-conviction moves. Designed for 1d timeframe to work in both bull and bear markets.
Target: 10-25 trades/year with disciplined entry conditions.
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
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50
    ema50_1w = np.full(len(close_1w), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1w)):
        if i == 50:
            ema50_1w[i] = np.mean(close_1w[0:51])
        else:
            ema50_1w[i] = close_1w[i] * k + ema50_1w[i-1] * (1 - k)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (gain[i] + 13 * avg_gain[i-1]) / 14
            avg_loss[i] = (loss[i] + 13 * avg_loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike and weekly uptrend (price > weekly EMA50)
            if (rsi[i] < 30 and vol_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with volume spike and weekly downtrend (price < weekly EMA50)
            elif (rsi[i] > 70 and vol_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or weekly trend turns down
            if (rsi[i] >= 50 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or weekly trend turns up
            if (rsi[i] <= 50 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Extremes_WeeklyTrend_Filtered"
timeframe = "1d"
leverage = 1.0