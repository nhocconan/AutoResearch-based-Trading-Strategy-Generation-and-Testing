#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h volume confirmation and 1d ADX regime filter.
# Long when price breaks above R1 AND 12h volume > 1.5x 20-period average AND 1d ADX > 20 (trending regime).
# Short when price breaks below S1 AND 12h volume > 1.5x 20-period average AND 1d ADX > 20 (trending regime).
# Uses discrete position size 0.25. Camarilla pivot levels provide intraday support/resistance that often act as breakout levels in trending markets.
# Volume surge confirms institutional participation. ADX filter ensures we only trade when higher timeframe is trending (avoiding false breakouts in ranges).
# Designed to work in both bull (breakout continuations) and bear (breakdown continuations) markets.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Typical Price for Camarilla calculation ===
    typical_price = (high + low + close) / 3.0
    
    # Get 1d data once before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate pivot and levels from previous 1d bar
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    R1_1d = pivot_1d + (range_1d * 1.1 / 12)
    S1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align 1d Camarilla levels to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Get 12h data once before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for volume MA
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (1.5 * vol_ma_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Get 1d data once before loop for ADX regime filter
    df_1d_adx = get_htf_data(prices, '1d')
    if len(df_1d_adx) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d_adx = df_1d_adx['high'].values
    low_1d_adx = df_1d_adx['low'].values
    close_1d_adx = df_1d_adx['close'].values
    
    # === 1d Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high_1d_adx).diff()
    tr2 = pd.Series(low_1d_adx).diff().abs()
    tr3 = pd.Series(close_1d_adx).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d_adx).diff()
    down_move = -pd.Series(low_1d_adx).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx_1d.values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d_adx, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r1_level = R1_aligned[i]
        s1_level = S1_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot level or volume spike ends
            if price <= pivot_1d[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot level or volume spike ends
            if price >= pivot_1d[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND 12h volume spike AND 1d ADX > 20 (trending regime)
            if price > r1_level and vol_spike and adx_val > 20:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND 12h volume spike AND 1d ADX > 20 (trending regime)
            elif price < s1_level and vol_spike and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hVolumeSpike_1dADX20_V1"
timeframe = "4h"
leverage = 1.0