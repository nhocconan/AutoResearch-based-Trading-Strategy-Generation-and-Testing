#!/usr/bin/env python3
"""
4h_VortexBreakout_1dTrend_Volume
Hypothesis: In trending markets (1d ADX > 25), Vortex indicator breakouts above/below thresholds align with trend direction and continue the move. In ranging markets (1d ADX < 25), mean reversion occurs at Vortex extremes. Volume confirmation (volume > 1.5x 20-period average) filters false signals. Designed for 20-40 trades/year per symbol to avoid fee drag while capturing both range reversals and trend continuations. Works in both bull and bear markets by adapting to regime.
"""

name = "4h_VortexBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX and Vortex
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d ADX for regime detection (14 period) ---
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
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d Vortex Indicator (14 period) ---
    # True Range already calculated as 'tr'
    # Vortex Up Movement
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    # Vortex Down Movement
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    
    # Sum over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Vortex Indicator
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Align Vortex and ADX to 4h
    vi_plus_4h = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_4h = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for ADX and Vortex
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vi_plus_4h[i]) or 
            np.isnan(vi_minus_4h[i]) or np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Check stoploss (2.0x ATR from estimate)
                atr_est = np.abs(high_4h[i] - low_4h[i])  # rough 4h ATR estimate
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_1d_aligned[i] < 25
        is_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for entries based on regime
            if is_range and vol_confirm:
                # Mean reversion: extreme Vortex readings
                if vi_minus_4h[i] > 1.1 and vi_plus_4h[i] < 0.9:
                    # Strong down Vortex, weak up Vortex = potential long mean reversion
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_4h[i]
                elif vi_plus_4h[i] > 1.1 and vi_minus_4h[i] < 0.9:
                    # Strong up Vortex, weak down Vortex = potential short mean reversion
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_4h[i]
            elif is_trend and vol_confirm:
                # Trend following: Vortex crossover in trend direction
                if vi_plus_4h[i] > vi_minus_4h[i] and vi_plus_4h[i-1] <= vi_minus_4h[i-1]:
                    # VI+ crosses above VI- = bullish signal
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_4h[i]
                elif vi_minus_4h[i] > vi_plus_4h[i] and vi_minus_4h[i-1] <= vi_plus_4h[i-1]:
                    # VI- crosses above VI+ = bearish signal
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, take profit when Vortex normalizes
                    if vi_plus_4h[i] < 1.05 and vi_minus_4h[i] < 1.05:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: VI- becomes too strong
                    elif vi_minus_4h[i] > 1.2:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # is_trend
                    # In trend, trail with Vortex strength
                    if vi_plus_4h[i] < vi_minus_4h[i]:
                        # Vortex bearish crossover
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: VI- becomes dominant
                    elif vi_minus_4h[i] > 1.3:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, take profit when Vortex normalizes
                    if vi_plus_4h[i] < 1.05 and vi_minus_4h[i] < 1.05:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: VI+ becomes too strong
                    elif vi_plus_4h[i] > 1.2:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # is_trend
                    # In trend, trail with Vortex strength
                    if vi_minus_4h[i] < vi_plus_4h[i]:
                        # Vortex bullish crossover
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: VI+ becomes dominant
                    elif vi_plus_4h[i] > 1.3:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals