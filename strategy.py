#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume spike and 1d ADX trend filter.
# Long when price crosses above Camarilla S1 pivot level AND 1d volume > 1.5x 20-period average AND ADX(14) > 20.
# Short when price crosses below Camarilla R1 pivot level AND 1d volume > 1.5x 20-period average AND ADX(14) > 20.
# Exit when price crosses back to the opposite pivot level (S1 for long exit, R1 for short exit).
# Uses Camarilla pivot points from daily data for structural support/resistance.
# Volume and ADX filters ensure trades occur in trending, high-volume environments.
# Designed for low-frequency, high-conviction trades to minimize fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "12h_Camarilla_S1R1_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla pivots, volume, and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Shift by 1 to use previous day's data for today's pivot calculation
    high_d_prev = np.roll(high_d, 1)
    low_d_prev = np.roll(low_d, 1)
    close_d_prev = np.roll(close_d, 1)
    
    # First day has no previous data
    high_d_prev[0] = high_d[0]
    low_d_prev[0] = low_d[0]
    close_d_prev[0] = close_d[0]
    
    # Camarilla pivot calculation
    range_prev = high_d_prev - low_d_prev
    camarilla_s1 = close_d_prev + 1.1 * range_prev * 1.0 / 12
    camarilla_r1 = close_d_prev + 1.1 * range_prev * 11.0 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_d, camarilla_r1)
    
    # Daily volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Daily ADX(14) for trend strength
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Avoid division by zero
    di_sum = plus_di + minus_di
    dx = np.where(di_sum != 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
    
    # ADX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Trend filter: ADX > 20
    trend_filter = adx_aligned > 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for daily indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price crosses above Camarilla S1, volume filter, trending market
            long_cond = (close[i] > camarilla_s1_aligned[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price crosses below Camarilla R1, volume filter, trending market
            short_cond = (close[i] < camarilla_r1_aligned[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla S1
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla R1
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals