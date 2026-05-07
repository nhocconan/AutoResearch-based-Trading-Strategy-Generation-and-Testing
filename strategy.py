#!/usr/bin/env python3
name = "1d_1w_TurtleBreakout"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR for position sizing and stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 20-day high in weekly uptrend
            if close[i] > high_20[i] and ema_10_aligned[i] > ema_10_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low in weekly downtrend
            elif close[i] < low_20[i] and ema_10_aligned[i] < ema_10_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to 20-day low or weekly trend reverses
            if close[i] < low_20[i] or ema_10_aligned[i] < ema_10_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to 20-day high or weekly trend reverses
            if close[i] > high_20[i] or ema_10_aligned[i] > ema_10_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Turtle-style breakout on daily timeframe with weekly trend filter
# - Enter long when price breaks above 20-day high during weekly uptrend (EMA10 rising)
# - Enter short when price breaks below 20-day low during weekly downtrend (EMA10 falling)
# - Exit when price returns to 20-day low/high or weekly trend reverses
# - Position size 0.25 limits drawdown during adverse moves (e.g., 2022 crash)
# - Weekly trend filter ensures we only trade in the direction of higher timeframe momentum
# - Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
# - Low trade frequency (~10-25/year) minimizes fee drag
# - Uses actual Donchian breakouts (proven effective) with proper trend alignment
# - Avoids overtrading by requiring clear weekly trend alignment for entries