#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Bull regime: ADX(1d) > 25 AND Bull Power > 0 → long on 6h Bull Power crossing above zero
    # Bear regime: ADX(1d) > 25 AND Bear Power < 0 → short on 6h Bear Power crossing below zero
    # Exit when power crosses zero opposite direction
    # Uses Elder Ray (Bull/Bear Power) to measure trend strength relative to EMA13
    # ADX regime filter ensures we only trade strong trends, avoiding whipsaws in ranging markets
    # Works in bull (continuation longs) and bear (continuation shorts) by adapting to regime
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe and Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for ADX regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for 6h (Elder Ray base)
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components for 6h
    bull_power_6h = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power_6h = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Calculate ADX for 1d (regime filter)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first values
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        dm_plus_smooth[atr_period-1] = np.mean(dm_plus[:atr_period])
        dm_minus_smooth[atr_period-1] = np.mean(dm_minus[:atr_period])
        
        # Wilder's smoothing for remaining values
        for i in range(atr_period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            dm_plus_smooth[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_smooth[i-1]
            dm_minus_smooth[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_smooth[i-1]
    
    # Avoid division by zero
    dm_plus_smooth = np.where(dm_plus_smooth == 0, 1e-10, dm_plus_smooth)
    dm_minus_smooth = np.where(dm_minus_smooth == 0, 1e-10, dm_minus_smooth)
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_safe
    di_minus = 100 * dm_minus_smooth / atr_safe
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = np.zeros_like(dx)
    
    # Wilder's smoothing for ADX
    if len(dx) >= atr_period:
        adx[atr_period-1] = np.mean(dx[:atr_period])
        for i in range(atr_period, len(dx)):
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align 6h Elder Ray components to 6h timeframe (no alignment needed, but for consistency)
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema13_6h[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        strong_trend = adx_aligned[i] > 25
        bull_regime = strong_trend and (bull_power_6h_aligned[i] > 0)
        bear_regime = strong_trend and (bear_power_6h_aligned[i] < 0)
        
        # Entry conditions: power crossing zero in direction of regime
        bull_cross_up = (bull_power_6h_aligned[i-1] <= 0 and bull_power_6h_aligned[i] > 0)
        bear_cross_down = (bear_power_6h_aligned[i-1] >= 0 and bear_power_6h_aligned[i] < 0)
        
        long_entry = bull_regime and bull_cross_up and position != 1
        short_entry = bear_regime and bear_cross_down and position != -1
        
        # Exit conditions: power crossing zero opposite direction
        exit_long = (position == 1 and bear_power_6h_aligned[i] > 0)  # Bear Power positive = exit long
        exit_short = (position == -1 and bull_power_6h_aligned[i] < 0)  # Bull Power negative = exit short
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0