#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND 1w ADX > 25 AND volume > 1.5 * average volume.
# Short when price breaks below Donchian lower AND 1w ADX > 25 AND volume > 1.5 * average volume.
# Uses discrete position size 0.25. Donchian provides clear breakout levels, ADX filters for trending markets only.
# Weekly ADX ensures alignment with higher timeframe trend strength to avoid false breakouts in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data once before loop for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX (14-period) ===
    # True Range
    tr1 = pd.Series(high_1w).rolling(window=1).max().values - pd.Series(low_1w).rolling(window=1).min().values
    tr2 = abs(pd.Series(high_1w).rolling(window=1).max().values - pd.Series(close_1w).shift(1).rolling(window=1).min().values)
    tr3 = abs(pd.Series(low_1w).rolling(window=1).min().values - pd.Series(close_1w).shift(1).rolling(window=1).max().values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = pd.Series(high_1w).diff().values
    down_move = -pd.Series(low_1w).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di_14 = 100 * plus_dm_14 / atr_1w
    minus_di_14 = 100 * minus_dm_14 / atr_1w
    
    # DX and ADX
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 + 14 + 14 periods for Donchian and ADX)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_1w_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian lower (breakdown)
            if price < donchian_lower[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian upper (breakout)
            if price > donchian_upper[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5 * average volume
            volume_confirmed = vol > 1.5 * avg_vol
            # ADX filter: trending market (ADX > 25)
            trending = adx_val > 25
            
            # LONG: Price breaks above Donchian upper AND volume confirmed AND trending
            if price > donchian_upper[i] and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND volume confirmed AND trending
            elif price < donchian_lower[i] and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1wADX_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0