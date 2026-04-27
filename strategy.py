#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA34 trend filter and choppiness regime.
In trending markets (price > weekly EMA34 for longs, < EMA34 for shorts), R1/S1 breakouts capture momentum.
Choppiness filter avoids false signals in ranging markets. Discrete position sizing (0.25) limits drawdown.
Targets 30-100 trades over 4 years on 1d timeframe. Works in both bull and bear via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Camarilla and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 1d (no shift needed as already 1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w EMA34 trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Choppiness regime: CHOP(14) < 61.8 = trending (favor breakouts)
    # True range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Positive and negative directional movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / np.where(atr_smooth != 0, atr_smooth, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.where(atr_smooth != 0, atr_smooth, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    chop = 100 * np.sqrt(np.log14(14) / np.log(2)) * np.sqrt(pd.Series(dx).rolling(window=14, min_periods=14).mean().values / 100)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position
    
    # Warmup: need 1d shift, EMA34, chop
    start_idx = max(30, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        
        # Only trade in trending regime (CHOP < 61.8)
        if chop_val >= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entry: Camarilla breakout with EMA alignment
            long_condition = (close_val > r1_val and close_val > ema_val)
            short_condition = (close_val < s1_val and close_val < ema_val)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price re-enters Camarilla range (below S1) OR loses EMA alignment
            if close_val < s1_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Camarilla range (above R1) OR loses EMA alignment
            if close_val > r1_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter"
timeframe = "1d"
leverage = 1.0