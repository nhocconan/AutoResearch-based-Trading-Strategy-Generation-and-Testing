#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h Williams %R overbought/oversold + 12h ADX trend filter + volume confirmation.
# Williams %R(14) identifies overextended moves: > -20 = overbought (short), < -80 = oversold (long).
# ADX(14) > 25 filters for trending conditions to avoid whipsaws in ranging markets.
# Volume confirmation ensures institutional participation.
# Designed for 12-37 trades/year to minimize fee drag while capturing mean reversion in trends.
# Works in bull/bear markets by combining momentum overextension with trend strength filtering.

name = "6h_12h_williams_r_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        (highest_high - close_12h) / (highest_high - lowest_low) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where(
        (high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]),
        np.maximum(high_12h[1:] - high_12h[:-1], 0),
        0
    )
    dm_minus = np.where(
        (low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]),
        np.maximum(low_12h[:-1] - low_12h[1:], 0),
        0
    )
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # skip first NaN
        # Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            else:
                result[i] = np.nan
        return result
    
    atr = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where(
        (di_plus + di_minus) != 0,
        np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100,
        0
    )
    adx = smooth_wilder(dx, 14)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    vol_avg_20 = np.full_like(volume_12h, np.nan, dtype=float)
    for i in range(19, len(volume_12h)):
        vol_avg_20[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align 12h indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 12h average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Williams %R signals
        williams_oversold = williams_r_aligned[i] < -80  # Oversold - long signal
        williams_overbought = williams_r_aligned[i] > -20  # Overbought - short signal
        
        # ADX trend filter: only trade when trending (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # Entry logic: Williams extremes only in trending markets
        long_signal = williams_oversold and trending and vol_filter
        short_signal = williams_overbought and trending and vol_filter
        
        # Exit when Williams %R returns to neutral range (-50 ± 15)
        exit_long = williams_r_aligned[i] > -65  # Exit long when less oversold
        exit_short = williams_r_aligned[i] < -35  # Exit short when less overbought
        
        # Update position and signals
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals