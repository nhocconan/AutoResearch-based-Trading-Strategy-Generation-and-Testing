#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Exponential Moving Average Crossover with 1-week ADX trend filter and 1-day volume confirmation
# Uses 12/26 EMA crossovers for momentum signals, weekly ADX(14) > 25 to ensure trending markets,
# and daily volume > 1.5x 20-period average to confirm institutional participation.
# Works in both bull and bear markets by only taking trades when ADX confirms strong trend.
# Target: 20-30 trades/year to minimize fee decay while capturing strong trending moves.
# Focus on BTC/ETH pairs with proven trend-following edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 12/26 EMA on 6h for crossover signals
    fast_len = 12
    slow_len = 26
    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    
    if n >= slow_len:
        # Initialize EMAs
        ema_fast[fast_len-1] = np.mean(close[:fast_len])
        ema_slow[slow_len-1] = np.mean(close[:slow_len])
        
        # Calculate multipliers
        multiplier_fast = 2 / (fast_len + 1)
        multiplier_slow = 2 / (slow_len + 1)
        
        # Calculate EMAs
        for i in range(fast_len, n):
            ema_fast[i] = (close[i] * multiplier_fast) + (ema_fast[i-1] * (1 - multiplier_fast))
        for i in range(slow_len, n):
            ema_slow[i] = (close[i] * multiplier_slow) + (ema_slow[i-1] * (1 - multiplier_slow))
    
    # Calculate ADX on 1w for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    n_1w = len(high_1w)
    
    # Calculate True Range
    tr = np.zeros(n_1w)
    for i in range(n_1w):
        if i == 0:
            tr[i] = high_1w[i] - low_1w[i]
        else:
            tr[i] = max(high_1w[i] - low_1w[i], 
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n_1w)
    minus_dm = np.zeros(n_1w)
    for i in range(1, n_1w):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (period=14)
    adx_len = 14
    atr = np.zeros(n_1w)
    plus_dm_smooth = np.zeros(n_1w)
    minus_dm_smooth = np.zeros(n_1w)
    
    if n_1w >= adx_len:
        # Initial values
        atr[adx_len-1] = np.mean(tr[:adx_len])
        plus_dm_smooth[adx_len-1] = np.mean(plus_dm[:adx_len])
        minus_dm_smooth[adx_len-1] = np.mean(minus_dm[:adx_len])
        
        # Wilder's smoothing
        for i in range(adx_len, n_1w):
            atr[i] = (atr[i-1] * (adx_len - 1) + tr[i]) / adx_len
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (adx_len - 1) + plus_dm[i]) / adx_len
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (adx_len - 1) + minus_dm[i]) / adx_len
    
    # Calculate DI and DX
    plus_di = np.zeros(n_1w)
    minus_di = np.zeros(n_1w)
    dx = np.zeros(n_1w)
    
    for i in range(adx_len, n_1w):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(n_1w)
    if n_1w >= 2 * adx_len - 1:
        adx[2*adx_len-2] = np.mean(dx[adx_len-1:2*adx_len-1])
        for i in range(2*adx_len-1, n_1w):
            adx[i] = (adx[i-1] * (adx_len - 1) + dx[i]) / adx_len
    
    # Calculate 20-period average volume on 1d for spike detection
    vol_ma_1d = np.full(len(df_1d), np.nan)
    vol_period = 20
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_period:i])
    
    # Align HTF indicators to 6s timeframe
    ema_fast_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_slow)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period - ensure all indicators are ready
    start_idx = max(slow_len, adx_len*2, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_fast_aligned[i]) or 
            np.isnan(ema_slow_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        volume_confirmation = vol_ratio > 1.5
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Fast EMA crosses above Slow EMA with trend and volume
            if ema_fast_aligned[i] > ema_slow_aligned[i] and strong_trend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Fast EMA crosses below Slow EMA with trend and volume
            elif ema_fast_aligned[i] < ema_slow_aligned[i] and strong_trend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Fast EMA crosses below Slow EMA or trend weakens
            if ema_fast_aligned[i] < ema_slow_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Fast EMA crosses above Slow EMA or trend weakens
            if ema_fast_aligned[i] > ema_slow_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA_Crossover_1wADX_1dVolume"
timeframe = "6h"
leverage = 1.0