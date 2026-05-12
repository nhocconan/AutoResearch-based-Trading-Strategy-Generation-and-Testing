#!/usr/bin/env python3
"""
4h EMA_Crossover + Volume + ADX Trend Filter
Hypothesis: EMA crossover captures momentum, volume confirms institutional interest,
and ADX > 25 filters for trending markets. This combination reduces false signals
in ranging markets while capturing sustained moves in both bull and bear cycles.
Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""
name = "4h_EMA_Crossover_Volume_ADX"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === EMA Fast (9) and Slow (21) ===
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === ADX (14) for trend strength ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_sum
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx, index=range(len(dx))).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Fast EMA above Slow EMA + ADX > 25 (trending) + volume spike
            if (ema_fast[i] > ema_slow[i] and 
                adx[i] > 25 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fast EMA below Slow EMA + ADX > 25 (trending) + volume spike
            elif (ema_fast[i] < ema_slow[i] and 
                  adx[i] > 25 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Fast EMA below Slow EMA OR ADX < 20 (no trend)
            if ema_fast[i] < ema_slow[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fast EMA above Slow EMA OR ADX < 20 (no trend)
            if ema_fast[i] > ema_slow[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals