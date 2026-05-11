#!/usr/bin/env python3
"""
12h_Supertrend_1dTrend_Filtered
Hypothesis: Use Supertrend on 12h for trend direction, filtered by 1d ADX to avoid counter-trend trades. In strong trends (1d ADX > 25), follow 12h Supertrend. In weak trends (1d ADX < 25), reduce position size to avoid whipsaw. Uses volume confirmation (volume > 1.3x 20-period average) to filter false signals. Designed for 15-30 trades/year per symbol to minimize fee drift while capturing major trends. Works in both bull and bear markets by adapting to trend strength.
"""

name = "12h_Supertrend_1dTrend_Filtered"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d ADX for trend strength filter (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 12h Supertrend (ATR=10, multiplier=3.0) ---
    # True Range for 12h
    tr1_12h = np.abs(high - low)
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + 3.0 * atr_12h
    basic_lb = (high + low) / 2 - 3.0 * atr_12h
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    
    for i in range(len(close)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend direction
    supertrend = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = 1.0  # start long
        else:
            if close[i] > final_ub[i-1]:
                supertrend[i] = 1.0
            elif close[i] < final_lb[i-1]:
                supertrend[i] = -1.0
            else:
                supertrend[i] = supertrend[i-1]
    
    supertrend_aligned = supertrend  # already on 12h timeframe
    
    # --- Volume confirmation ---
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30  # for ADX and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                # Simple trailing stop: reverse if Supertrend flips
                if position == 1 and supertrend_aligned[i] == -1:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and supertrend_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine trend strength: ADX < 25 = weak, ADX > 25 = strong
        is_strong_trend = adx_12h_aligned[i] > 25
        is_weak_trend = adx_12h_aligned[i] < 25
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        if position == 0:
            # Look for entries based on Supertrend and trend strength
            if vol_confirm:
                if supertrend_aligned[i] == 1:
                    if is_strong_trend:
                        signals[i] = 0.30  # full position in strong trend
                        position = 1
                    else:
                        signals[i] = 0.15  # half position in weak trend
                        position = 1
                elif supertrend_aligned[i] == -1:
                    if is_strong_trend:
                        signals[i] = -0.30  # full position in strong trend
                        position = -1
                    else:
                        signals[i] = -0.15  # half position in weak trend
                        position = -1
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if supertrend_aligned[i] == -1:
                    # Supertrend flipped to down - exit
                    signals[i] = 0.0
                    position = 0
                else:
                    # Still in uptrend - hold
                    signals[i] = 0.30 if is_strong_trend else 0.15
            elif position == -1:
                # Short position management
                if supertrend_aligned[i] == 1:
                    # Supertrend flipped to up - exit
                    signals[i] = 0.0
                    position = 0
                else:
                    # Still in downtrend - hold
                    signals[i] = -0.30 if is_strong_trend else -0.15
    
    return signals