#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Range Breakout with 1-week Volume Confirmation and ADX Trend Filter.
# Long when price breaks above Daily Range High (20-day high) AND 1w volume > 1.5x 20-week average AND ADX(14) > 25 (trending market).
# Short when price breaks below Daily Range Low (20-day low) AND 1w volume > 1.5x 20-week average AND ADX(14) > 25.
# Exit when price crosses back inside the 20-day range.
# Uses 1d timeframe as specified, with 1w volume and ADX for higher timeframe context.
# Target: 30-100 total trades over 4 years (7-25/year) with controlled frequency to avoid fee drag.
# Works in both bull and bear markets by only trading in strong trends (ADX > 25) and using volatility-based breakouts.

name = "1d_DailyRangeBreakout_1wVolume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for volume and ADX
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Daily Range: 20-day high/low
    period = 20
    range_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    range_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Weekly volume filter: current volume > 1.5x 20-week average
    volume_w = df_w['volume'].values
    vol_ma20_w = pd.Series(volume_w).rolling(window=20, min_periods=20).mean().values
    volume_filter_w = volume_w > (1.5 * vol_ma20_w)
    volume_filter = align_htf_to_ltf(prices, df_w, volume_filter_w)
    
    # Weekly ADX(14) for trend strength
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # True Range
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_w[0] - low_w[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_w - np.roll(high_w, 1)) > (np.roll(low_w, 1) - low_w), 
                       np.maximum(high_w - np.roll(high_w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_w, 1) - low_w) > (high_w - np.roll(high_w, 1)), 
                        np.maximum(np.roll(low_w, 1) - low_w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_w, adx)
    
    # Trend filter: ADX > 25
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(range_high[i]) or np.isnan(range_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above range high, volume filter, trending market
            long_cond = (close[i] > range_high[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below range low, volume filter, trending market
            short_cond = (close[i] < range_low[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below range low
            if close[i] < range_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above range high
            if close[i] > range_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals