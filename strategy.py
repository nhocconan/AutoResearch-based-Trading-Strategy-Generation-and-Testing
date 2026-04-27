#!/usr/bin/env python3
"""
4h_RollingBreakout_VolumeConfirm_ADXFilter_v1
Hypothesis: Rolling 20-period high/low breakouts with volume confirmation and ADX trend filter
capture strong momentum moves while avoiding false breakouts in ranging markets. Uses 4h timeframe
to balance trade frequency and signal quality. Target: 20-40 trades/year for low fee drag.
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
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean()
    
    atr = tr_smooth.values
    plus_di = 100 * plus_dm_smooth.values / np.where(atr == 0, 1, atr)
    minus_di = 100 * minus_dm_smooth.values / np.where(atr == 0, 1, atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Rolling 20-period high/low for breakout
    lookback = 20
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ADX, roll_high/low, volume MA
    start_idx = max(lookback, 34)  # ADX needs ~34, roll needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: break above rolling high, ADX > 25, volume confirmation
            if close[i] > roll_high[i] and adx_val > 25 and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: break below rolling low, ADX > 25, volume confirmation
            elif close[i] < roll_low[i] and adx_val > 25 and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: break below rolling low or ADX weakens
            if close[i] < roll_low[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above rolling high or ADX weakens
            if close[i] > roll_high[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RollingBreakout_VolumeConfirm_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0