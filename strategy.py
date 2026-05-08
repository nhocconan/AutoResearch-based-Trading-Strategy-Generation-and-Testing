#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with weekly volume confirmation and monthly ADX trend filter.
# Long when price breaks above prior bearish fractal AND weekly volume > 1.2x 4-week average AND monthly ADX > 25.
# Short when price breaks below prior bullish fractal AND weekly volume > 1.2x 4-week average AND monthly ADX > 25.
# Exit when price crosses back inside the prior fractal pair (between bullish and bearish fractal).
# Williams Fractals provide natural support/resistance levels, volume confirms conviction, ADX ensures trending environment.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_Williams_Fractal_Volume_ADX"
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
    
    # Weekly data for volume filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 4:
        return np.zeros(n)
    
    # Daily data for Williams Fractal calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 5:
        return np.zeros(n)
    
    # Monthly data for ADX trend filter
    df_m = get_htf_data(prices, '1M')
    if len(df_m) < 14:
        return np.zeros(n)
    
    # Williams Fractal calculation (5-bar window: bar is fractal if two lower highs on each side and two higher lows)
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    high_arr = df_d['high'].values
    low_arr = df_d['low'].values
    
    bearish_fractal = np.full(len(high_arr), np.nan)
    bullish_fractal = np.full(len(low_arr), np.nan)
    
    # Calculate fractals (need 2 bars on each side)
    for i in range(2, len(high_arr) - 2):
        # Bearish fractal: current high is highest among 5 bars
        if (high_arr[i] > high_arr[i-2] and high_arr[i] > high_arr[i-1] and 
            high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
            bearish_fractal[i] = high_arr[i]
        # Bullish fractal: current low is lowest among 5 bars
        if (low_arr[i] < low_arr[i-2] and low_arr[i] < low_arr[i-1] and 
            low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
            bullish_fractal[i] = low_arr[i]
    
    # Williams fractals need 2 extra bars for confirmation (pattern complete after 2nd bar after center)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_d, bullish_fractal, additional_delay_bars=2)
    
    # Weekly volume filter: current volume > 1.2x 4-week average
    vol_ma4w = pd.Series(df_w['volume'].values).rolling(window=4, min_periods=4).mean().values
    # Align weekly volume MA to 12h timeframe
    vol_ma4w_aligned = align_htf_to_ltf(prices, df_w, vol_ma4w)
    volume_filter = volume > (1.2 * vol_ma4w_aligned)
    
    # Monthly ADX trend filter
    high_m = df_m['high'].values
    low_m = df_m['low'].values
    close_m = df_m['close'].values
    
    # True Range
    tr1 = high_m - low_m
    tr2 = np.abs(high_m - np.roll(close_m, 1))
    tr3 = np.abs(low_m - np.roll(close_m, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First TR is just high-low (no previous close)
    tr[0] = high_m[0] - low_m[0]
    
    # Directional Movement
    dm_plus = np.where((high_m - np.roll(high_m, 1)) > (np.roll(low_m, 1) - low_m), 
                       np.maximum(high_m - np.roll(high_m, 1), 0), 0)
    dm_minus = np.where((np.roll(low_m, 1) - low_m) > (high_m - np.roll(high_m, 1)), 
                        np.maximum(np.roll(low_m, 1) - low_m, 0), 0)
    # First DM values are 0 (no previous period)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing: alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average of first period)
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing for remaining values
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1e-10)
    minus_di = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period-1])  # First ADX after 2*period
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align monthly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_m, adx)
    
    # Trend filter: ADX > 25 indicates strong trend
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 4*4, 2*14)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal, volume filter, trend filter
            long_cond = (close[i] > bearish_fractal_confirmed[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below bullish fractal, volume filter, trend filter
            short_cond = (close[i] < bullish_fractal_confirmed[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below bullish fractal
            if close[i] < bullish_fractal_confirmed[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above bearish fractal
            if close[i] > bearish_fractal_confirmed[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals