#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX(20) trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND 12h ADX > 20 (trending up) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower AND 12h ADX > 20 (trending down) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 12h ADX ensures higher timeframe trend alignment (avoiding counter-trend whipsaws),
# volume spike confirms institutional participation. Designed to catch strong trends in both bull (breakouts up) and bear (breakdowns down) markets.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Donchian upper = highest high over 20 periods
    # Donchian lower = lowest low over 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: ADX(20) for trend filter ===
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    
    # Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = -pd.Series(low_12h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    adx_values = adx.values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 40 periods needed for Donchian, 40 for ADX, 20 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        donch_up = donchian_upper[i]
        donch_low = donchian_lower[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian lower or volume spike ends
            if price <= donch_low or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian upper or volume spike ends
            if price >= donch_up or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND 12h ADX > 20 (uptrend) AND volume spike
            if price > donch_up and adx_val > 20 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND 12h ADX > 20 (downtrend) AND volume spike
            elif price < donch_low and adx_val > 20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hADX20_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0