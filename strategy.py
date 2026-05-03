#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Long: Close breaks above Donchian upper AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Short: Close breaks below Donchian lower AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or ADX < 20 (range) or volume drops.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# 12h timeframe reduces trade frequency vs 4h, minimizing fee drag. 1w ADX provides strong trend filter
# to avoid whipsaws in ranging markets. Volume confirmation ensures breakouts have participation.
# Works in bull via long signals and bear via short signals when aligned with strong weekly trend.

name = "12h_Donchian20_1wADX25_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # True Range
    tr1 = df_1w_high - df_1w_low
    tr2 = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3 = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((df_1w_high - np.roll(df_1w_high, 1)) > (np.roll(df_1w_low, 1) - df_1w_low), 
                       np.maximum(df_1w_high - np.roll(df_1w_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1w_low, 1) - df_1w_low) > (df_1w_high - np.roll(df_1w_high, 1)), 
                        np.maximum(np.roll(df_1w_low, 1) - df_1w_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1w ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian channels (20-period) on 12h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND trending AND volume spike
            if close_val > donchian_upper[i] and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower AND trending AND volume spike
            elif close_val < donchian_lower[i] and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian lower OR trend weakens (ADX < 20) OR volume drops
            if close_val < donchian_lower[i] or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian upper OR trend weakens (ADX < 20) OR volume drops
            if close_val > donchian_upper[i] or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals