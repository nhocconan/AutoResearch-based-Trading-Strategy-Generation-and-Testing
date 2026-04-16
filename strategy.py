#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly ADX(14) trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly ADX > 25 (strong trend) AND 1d volume > 1.5x 20-day average.
# Short when price breaks below 20-day low AND weekly ADX > 25 AND 1d volume > 1.5x 20-day average.
# Uses discrete position size 0.30. Donchian captures momentum, ADX filters choppy markets, volume confirms participation.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get weekly data once before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = pd.Series(high_1w).rolling(window=2).max().values - pd.Series(low_1w).rolling(window=2).min().values
    tr2 = abs(pd.Series(high_1w).values - pd.Series(close_1w).shift(1).values)
    tr3 = abs(pd.Series(low_1w).values - pd.Series(close_1w).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w).values - pd.Series(high_1w).shift(1).values
    down_move = pd.Series(low_1w).shift(1).values - pd.Series(low_1w).values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for Donchian/volume MA, 14*3 for ADX)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        hh = highest_high[i]
        ll = lowest_low[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below 10-day low (trailing stop) or ADX weakens
            if price < lowest_low[i] or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above 10-day high (trailing stop) or ADX weakens
            if price > highest_high[i] or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 20-day high AND ADX > 25 (strong trend) AND volume spike
            if price > hh and adx_val > 25 and vol_spike:
                signals[i] = 0.30
                position = 1
            
            # SHORT: Price breaks below 20-day low AND ADX > 25 AND volume spike
            elif price < ll and adx_val > 25 and vol_spike:
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "1d_Donchian20_1wADX25_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0