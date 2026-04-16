#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w ADX(14) regime filter.
# Long when price breaks above Donchian(20) upper band AND 1d volume > 1.5x 20-period average AND 1w ADX(14) > 25 (trending regime).
# Short when price breaks below Donchian(20) lower band AND 1d volume > 1.5x 20-period average AND 1w ADX(14) > 25 (trending regime).
# Uses discrete position size 0.25. Donchian captures structural breaks, volume confirms participation, 1w ADX ensures we only trade in higher timeframe trending markets (avoiding chop).
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets during trending conditions.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Get 1w data once before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = -pd.Series(low_1w).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1w_values = adx_1w.values
    
    # Align 1w ADX to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for Donchian/volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        vol_spike = volume_spike[i]
        vol_spike_1d = volume_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        adx_val = adx_1w_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midpoint of Donchian channel or volume spike ends
            midpoint = (highest + lowest) / 2
            if price <= midpoint or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midpoint of Donchian channel or volume spike ends
            midpoint = (highest + lowest) / 2
            if price >= midpoint or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian(20) upper band AND 1d volume spike AND 1w ADX > 25 (trending regime)
            if price > highest and vol_spike and vol_spike_1d and adx_val > 25:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian(20) lower band AND 1d volume spike AND 1w ADX > 25 (trending regime)
            elif price < lowest and vol_spike and vol_spike_1d and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_1wADX25_V1"
timeframe = "4h"
leverage = 1.0