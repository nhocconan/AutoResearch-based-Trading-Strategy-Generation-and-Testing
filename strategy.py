#!/usr/bin/env python3
# 4H_Vortex_Volume_Crossover_v1
# Hypothesis: Vortex Indicator crossovers combined with volume confirmation
# capture trend changes in both bull and bear markets. Vortex identifies
# directional movement strength, volume confirms conviction. Uses 4h timeframe
# with 1d ADX regime filter to avoid chop. Targets 25-50 trades/year for low
# frequency and minimal fee drag. Works in trends (trend follow) and ranges
# (mean reversion via Vortex crossovers at extremes).

name = "4H_Vortex_Volume_Crossover_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Vortex Indicator (14-period) on 4h ---
    # VM+ = |High - Prior Low|, VM- = |Low - Prior High|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])  # first period
    vm_minus[0] = vm_plus[0]
    
    # True Range
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    
    # Sum over 14 periods
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # --- 1d ADX (14-period) for regime filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # True Range for 1d
    tr1_1d = np.abs(high_1d - np.roll(low_1d, 1))
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Smoothed values
    tr_sum_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum_1d = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum_1d = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di_1d = 100 * (plus_dm_sum_1d / tr_sum_1d)
    minus_di_1d = 100 * (minus_dm_sum_1d / tr_sum_1d)
    
    # DX and ADX
    dx_denom = plus_di_1d + minus_di_1d
    dx_denom = np.where(dx_denom == 0, 1e-10, dx_denom)
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / dx_denom
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if ADX is NaN
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                # Hold position until ADX available
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Regime filter: only trade when ADX > 20 (trending market)
        if adx_1d_aligned[i] < 20:
            # In chop, exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Vortex crossover signals
        vi_plus_crossover = vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]
        vi_minus_crossover = vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]
        
        # Entry conditions
        long_entry = vi_plus_crossover and vol_spike[i]
        short_entry = vi_minus_crossover and vol_spike[i]
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on Vortex reverse crossover
            if position == 1:
                if vi_minus_crossover:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if vi_plus_crossover:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals