#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with weekly trend filter and volume confirmation
# KAMA adapts to market noise - faster in trends, slower in ranges
# Weekly trend filter ensures we only trade in direction of higher timeframe trend
# Volume confirmation filters false breakouts
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries
name = "1d_KAMA_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter - EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # ER over 10 periods
    er_num = np.abs(np.subtract(close[9:], np.append([np.nan]*9, close[:-9])))
    er_den = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np.sum, 'axis') else None
    
    # Simplified ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA + above weekly EMA34 + volume confirmation
            if (close[i] > kama[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + below weekly EMA34 + volume confirmation
            elif (close[i] < kama[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or below weekly EMA34
            if (close[i] < kama[i]) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or above weekly EMA34
            if (close[i] > kama[i]) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals