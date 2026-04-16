#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1h ADX trend filter.
# Long when price breaks above 4h Donchian upper band AND volume > 1.8x 20-period 1d average AND 1h ADX > 25 (trending market).
# Short when price breaks below 4h Donchian lower band AND volume > 1.8x 20-period 1d average AND 1h ADX > 25.
# Exit when price crosses the 4h Donchian midpoint (upper+lower)/2.
# Uses discrete position size 0.25. Designed to capture breakouts in trending markets (both bull and bear).
# Target: 100-180 total trades over 4 years (25-45/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) channels ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels: highest high and lowest low over 20 periods
    highest_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_4h = highest_4h
    lower_4h = lowest_4h
    midpoint_4h = (upper_4h + lower_4h) / 2
    
    # Align to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # === 1h Indicators: ADX > 25 (trending market filter) ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high).diff()
    dm_minus = pd.Series(low).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    trending = adx > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or np.isnan(midpoint_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trending[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = trending[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < midpoint_4h_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > midpoint_4h_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 4h upper band AND volume spike AND trending market
            if price > upper_4h_aligned[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below 4h lower band AND volume spike AND trending market
            elif price < lower_4h_aligned[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_1d_ADX1h_V1"
timeframe = "4h"
leverage = 1.0