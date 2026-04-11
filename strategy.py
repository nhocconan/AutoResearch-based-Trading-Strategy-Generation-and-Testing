#!/usr/bin/env python3
# 4h_1d_market_structure_v1
# Strategy: 4h market structure with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Combines higher timeframe trend direction (1d EMA) with lower timeframe
# market structure (swing highs/lows) and volume confirmation to capture
# trend continuation moves. Designed for low frequency (15-25 trades/year) to
# minimize fee drag while capturing significant moves in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_market_structure_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Swing points calculation (5-period lookback)
    # Swing high: current high is the highest in last 5 periods
    # Swing low: current low is the lowest in last 5 periods
    window = 5
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(window, n - window):
        # Check if current high is highest in window
        if high[i] == np.max(high[i-window:i+window+1]):
            swing_high[i] = True
        # Check if current low is lowest in window
        if low[i] == np.min(low[i-window:i+window+1]):
            swing_low[i] = True
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: break of swing point with volume spike and trend alignment
        long_entry = swing_high[i] and volume_spike[i] and uptrend
        short_entry = swing_low[i] and volume_spike[i] and downtrend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal
        elif position == 1 and not uptrend:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not downtrend:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals