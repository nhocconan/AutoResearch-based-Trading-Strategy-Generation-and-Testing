#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (12/26) with 4h trend filter (EMA50) and volume confirmation
# EMA crossover captures medium-term momentum with reasonable lag
# 4h EMA50 provides higher timeframe trend bias to avoid counter-trend trades
# Volume confirmation filters weak signals and confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) with disciplined entries
# Works in bull markets via momentum capture and in bear markets via trend filtering
name = "1h_EMACrossover_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # EMA crossover: 12/26
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 50, 20)  # EMA26, EMA50_4h, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: EMA12 > EMA26 + price > 4h EMA50 + volume confirmation
            if (ema_fast[i] > ema_slow[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: EMA12 < EMA26 + price < 4h EMA50 + volume confirmation
            elif (ema_fast[i] < ema_slow[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if EMA crossover reverses or price breaks below 4h EMA50
            if (ema_fast[i] < ema_slow[i]) or (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if EMA crossover reverses or price breaks above 4h EMA50
            if (ema_fast[i] > ema_slow[i]) or (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals