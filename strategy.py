#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1
Concept: 12h Camarilla pivot breakout with daily volume confirmation and ADX trend filter.
- Long: Price breaks above R1 AND volume > 1.3x daily average AND ADX > 20 (trending)
- Short: Price breaks below S1 AND volume > 1.3x daily average AND ADX > 20 (trending)
- Exit: Price crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 12-37 trades/year (50-150 total over 4 years)
- Works in bull/bear: ADX ensures we trade only in trending conditions, avoiding chop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h: Close prices ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h: ADX for trend strength ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth > 0, tr_smooth, np.nan)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth > 0, tr_smooth, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Daily: Pivot points (Camarilla) ===
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    r1 = prev_close + range_ * 1.1 / 12
    s1 = prev_close - range_ * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Volume context ===
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma20_1d > 0, vol_ma20_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Ensure enough data for ADX
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        adx_val = adx[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_1d_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(adx_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 AND volume confirmation AND trending (ADX > 20)
            if close_val > r1_val and vol_ratio_1d_val > 1.3 and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND volume confirmation AND trending (ADX > 20)
            elif close_val < s1_val and vol_ratio_1d_val > 1.3 and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals