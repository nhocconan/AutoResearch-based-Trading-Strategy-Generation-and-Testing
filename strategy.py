#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dADX25_TrendFilter_VolumeConfirm
Hypothesis: 6h Donchian(20) breakouts with 1d ADX>25 trend filter and volume confirmation capture strong momentum moves in both bull and bear markets. The 1d ADX ensures we only trade when there is a strong daily trend, reducing whipsaws. Volume confirmation adds conviction. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) with min_periods"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(np.abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Movement
    up_move = pd.Series(high - high.shift(1))
    down_move = pd.Series(low.shift(1) - low)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX > 25 trend filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    strong_trend = adx_1d_aligned > 25
    
    # 6h Donchian(20) breakout levels
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + ADX (14+14)
    start_idx = max(donchian_period, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + strong 1d trend + volume confirmation
            long_breakout = curr_high > highest_high[i]
            short_breakout = curr_low < lowest_low[i]
            
            long_entry = long_breakout and strong_trend[i] and volume_confirm[i]
            short_entry = short_breakout and strong_trend[i] and volume_confirm[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Donchian lower band or trend weakens
            if curr_close < lowest_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian upper band or trend weakens
            if curr_close > highest_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dADX25_TrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0