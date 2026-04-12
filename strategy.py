#!/usr/bin/env python3

# 12h_1d_Three_Level_Breakout_V1
# Hypothesis: Breakouts beyond three standard deviations from 1-day mean price,
# combined with volume surge and ADX trend filter, capture strong momentum moves.
# Works in bull markets (upward breaks) and bear markets (downward breaks).
# Target: 20-35 trades per year (80-140 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Three_Level_Breakout_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for statistical levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # === STATISTICAL LEVELS (3 std dev from 20-day mean) ===
    # Use previous 20-day close to avoid look-ahead
    prev_close = np.roll(daily_close, 1)
    
    # Calculate 20-day mean and std dev of close prices
    mean_20 = np.full(len(prev_close), np.nan)
    std_20 = np.full(len(prev_close), np.nan)
    
    if len(prev_close) >= 20:
        # Initialize first valid value
        mean_20[19] = np.mean(prev_close[:20])
        std_20[19] = np.std(prev_close[:20])
        
        # Rolling calculation
        for i in range(20, len(prev_close)):
            # Update mean
            mean_20[i] = mean_20[i-1] + (prev_close[i] - prev_close[i-20]) / 20
            # Update variance using Welford's algorithm approximation
            # For simplicity, we recalculate std over window (acceptable for 20 period)
            window = prev_close[i-19:i+1]
            std_20[i] = np.std(window)
    
    # Calculate upper and lower bands (3 std dev)
    upper_band = mean_20 + (3 * std_20)
    lower_band = mean_20 - (3 * std_20)
    
    # Align to 12h timeframe
    upper_band_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    mean_20_12h = align_htf_to_ltf(prices, df_1d, mean_20)
    
    # === VOLUME FILTER (1.5x 20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_filter = volume > (vol_ma * 1.5)
    
    # === ADX TREND FILTER (14-period) ===
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan, dtype=np.float64)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial averages
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_values = calculate_adx(high, low, close, 14)
    adx_filter = adx_values > 20  # Only trade when trend is present
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i]) or
            np.isnan(mean_20_12h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume and trend confirmation
        break_above = close[i] > upper_band_12h[i] and vol_filter[i] and adx_filter[i]
        break_below = close[i] < lower_band_12h[i] and vol_filter[i] and adx_filter[i]
        
        # Exit when price returns to mean
        return_to_mean = np.abs(close[i] - mean_20_12h[i]) < (0.1 * std_20[i]) if not np.isnan(std_20[i]) else False
        
        # Signal logic
        if break_above and position != 1:
            position = 1
            signals[i] = 0.25
        elif break_below and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and return_to_mean:
            position = 0
            signals[i] = 0.0
        elif position == -1 and return_to_mean:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals