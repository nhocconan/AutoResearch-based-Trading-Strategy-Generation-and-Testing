# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Supertrend_DualTrendFilter_1dADX
Hypothesis: Supertrend on 6h with dual timeframe trend filter (6h EMA50 + 1d ADX>25).
Only takes long when both timeframes are bullish, short when both bearish.
Avoids whipsaws by requiring alignment across timeframes. Works in bull/bear by
following the dominant trend on multiple timeframes. Uses fixed position size
to control risk and minimize fee churn.
"""

name = "6h_Supertrend_DualTrendFilter_1dADX"
timeframe = "6h"
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
    
    # Calculate Supertrend on 6h
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[0:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (multiplier * tr[i] + (atr_period-1) * atr[i-1]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full_like(close, np.nan)
    dir = np.full_like(close, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, n):
        if np.isnan(atr[i]):
            continue
            
        if i == atr_period:
            supertrend[i] = lower_band[i]
            dir[i] = 1
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if close[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    dir[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    dir[i] = 1
            else:
                if close[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    dir[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    dir[i] = -1
    
    # Calculate 6h EMA50 for trend filter
    ema_period = 50
    ema_50 = np.full_like(close, np.nan)
    if len(close) >= ema_period:
        ema_50[ema_period-1] = np.mean(close[0:ema_period])
        for i in range(ema_period, len(close)):
            ema_50[i] = (close[i] * 2 + ema_50[i-1] * (ema_period-1)) / (ema_period+1)
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d
    adx_period = 14
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = np.full_like(tr_1d, np.nan)
    plus_dm_14 = np.full_like(plus_dm, np.nan)
    minus_dm_14 = np.full_like(minus_dm, np.nan)
    
    if len(tr_1d) >= adx_period:
        tr_14[adx_period-1] = np.sum(tr_1d[0:adx_period])
        plus_dm_14[adx_period-1] = np.sum(plus_dm[0:adx_period])
        minus_dm_14[adx_period-1] = np.sum(minus_dm[0:adx_period])
        
        for i in range(adx_period, len(tr_1d)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / adx_period) + tr_1d[i]
            plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1] / adx_period) + plus_dm[i]
            minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1] / adx_period) + minus_dm[i]
    
    # DI+ and DI-
    plus_di = np.full_like(tr_1d, np.nan)
    minus_di = np.full_like(tr_1d, np.nan)
    valid = (~np.isnan(tr_14)) & (tr_14 != 0)
    plus_di[valid] = 100 * plus_dm_14[valid] / tr_14[valid]
    minus_di[valid] = 100 * minus_dm_14[valid] / tr_14[valid]
    
    # DX and ADX
    dx = np.full_like(tr_1d, np.nan)
    dx_valid = (~np.isnan(plus_di)) & (~np.isnan(minus_di)) & ((plus_di + minus_di) != 0)
    dx[dx_valid] = 100 * np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / (plus_di[dx_valid] + minus_di[dx_valid])
    
    adx = np.full_like(tr_1d, np.nan)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.mean(dx[adx_period-1:2*adx_period-1])
        for i in range(adx_period, len(dx)):
            adx[i] = (dx[i] + (adx_period-1) * adx[i-1]) / adx_period
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align Supertrend and EMA50 to 6h (they're already on 6h, but align for consistency)
    supertrend_aligned = supertrend  # Already on 6h
    ema_50_aligned = ema_50  # Already on 6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, ema_period, adx_period*2)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Supertrend uptrend, price > EMA50, and ADX > 25 (strong trend)
            if (supertrend_aligned[i] < close[i] and  # Supertrend indicates uptrend
                close[i] > ema_50_aligned[i] and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: Supertrend downtrend, price < EMA50, and ADX > 25 (strong trend)
            elif (supertrend_aligned[i] > close[i] and  # Supertrend indicates downtrend
                  close[i] < ema_50_aligned[i] and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend flips to downtrend OR price < EMA50
            if supertrend_aligned[i] > close[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend flips to uptrend OR price > EMA50
            if supertrend_aligned[i] < close[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals