#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Weekly Pivot R2/S2 breakout with daily volume confirmation and ADX trend filter.
# Long when price breaks above Weekly R2 AND volume > 1.3x 20-period average AND ADX > 25 (trending).
# Short when price breaks below Weekly S2 AND volume > 1.3x 20-period average AND ADX > 25.
# Exit when price crosses back inside the Weekly H-L range.
# Weekly pivots provide strong institutional levels; volume confirms participation; ADX filters ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_WeeklyPivot_R2S2_Volume_ADX"
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
    
    # Weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points (R2, S2) from previous week's OHLC
    prev_close_w = df_w['close'].shift(1).values
    prev_high_w = df_w['high'].shift(1).values
    prev_low_w = df_w['low'].shift(1).values
    prev_range_w = prev_high_w - prev_low_w
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    r2_w = pivot_w + prev_range_w
    s2_w = pivot_w - prev_range_w
    
    # Align Weekly R2/S2 to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Daily volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    # ADX trend filter (14-period) on 12h data
    # ADX > 25 indicates strong trend
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr = np.where(np.arange(len(close)) == 0, high - low, tr2)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    atr_smooth = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: ADX > 25
    trend_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14*3)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Weekly R2, volume filter, trending market
            long_cond = (close[i] > r2_aligned[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below Weekly S2, volume filter, trending market
            short_cond = (close[i] < s2_aligned[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Weekly S2
            if close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Weekly R2
            if close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals