#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
Long when price breaks above 20-day high AND 1w EMA50 is rising.
Short when price breaks below 20-day low AND 1w EMA50 is falling.
Exit when price touches opposite Donchian level or 1w EMA50 reverses direction.
Uses 1w HTF for EMA50 trend to reduce whipsaws and capture major trends.
Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe.
Donchian channels from 1d: upper = max(high, 20), lower = min(low, 20).
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian (20), EMA50 (50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        upper = high_roll[i]
        lower = low_roll[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA50 rising
            if price > upper and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA50 falling
            elif price < lower and ema_falling:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower band OR EMA50 starts falling
                if price < lower or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper band OR EMA50 starts rising
                if price > upper or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0