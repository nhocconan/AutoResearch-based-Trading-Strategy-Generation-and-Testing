#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend with 1d EMA13 trend filter and volume confirmation
# Supertrend captures trends effectively using ATR-based dynamic bands
# 1d EMA13 provides higher timeframe bias to avoid counter-trend trades
# Volume confirmation filters weak breakouts and confirms strength
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_Supertrend_1dEMA13_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA13 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Supertrend calculation on 4h
    period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = np.zeros(n)
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Supertrend bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, n):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend direction
    supertrend = np.zeros(n)
    supertrend[0] = final_lb[0]
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > final_ub[i-1]:
            direction[i] = 1
        elif close[i] < final_lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction[i]) or np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend + above 1d EMA13 + volume confirmation
            if (direction[i] == 1 and 
                close[i] > ema_13_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend + below 1d EMA13 + volume confirmation
            elif (direction[i] == -1 and 
                  close[i] < ema_13_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Supertrend turns downtrend or breaks below 1d EMA13
            if (direction[i] == -1) or (close[i] < ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Supertrend turns uptrend or breaks above 1d EMA13
            if (direction[i] == 1) or (close[i] > ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals