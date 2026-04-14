#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX filter and volume confirmation
# Uses Donchian channel breakout for trend following, filtered by 1d ADX > 25 to ensure
# trending market conditions and volume > 1.5x 20-period average for confirmation.
# Position exits when price crosses opposite Donchian band or ADX falls below 20.
# Designed for low-frequency trading (target: 15-25 trades/year) to minimize fee drag
# Works in both bull and bear markets by only taking trades in direction of 1d trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average of first 14 periods)
    atr[13] = np.mean(tr[1:14])
    dm_plus_smooth[13] = np.mean(dm_plus[1:14])
    dm_minus_smooth[13] = np.mean(dm_minus[1:14])
    
    # Wilder's smoothing: today's value = (13/14)*yesterday + (1/14)*today
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX after 2*14 periods
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 12h timeframe (with 1-bar delay for completed 1d bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over 20 periods
    upper = np.full_like(high_12h, np.nan)
    for i in range(19, len(high_12h)):
        upper[i] = np.max(high_12h[i-19:i+1])
    
    # Lower band: lowest low over 20 periods
    lower = np.full_like(low_12h, np.nan)
    for i in range(19, len(low_12h)):
        lower[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian bands to 12h timeframe (no additional delay needed)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 28)  # Donchian(20) and ADX need ~28 periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_value = adx_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with strong trend and volume
            if price > upper_aligned[i] and adx_value > 25 and vol_ratio[i] > 1.5:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian band with strong trend and volume
            elif price < lower_aligned[i] and adx_value > 25 and vol_ratio[i] > 1.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower band OR trend weakens (ADX < 20)
            if price < lower_aligned[i] or adx_value < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper band OR trend weakens (ADX < 20)
            if price > upper_aligned[i] or adx_value < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_breakout_ADX_volume"
timeframe = "12h"
leverage = 1.0