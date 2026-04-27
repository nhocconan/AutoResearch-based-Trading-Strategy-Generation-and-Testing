#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
# Uses 4h price channel breakout (Donchian) as primary signal, confirmed by volume > 1.5x average
# and 1d EMA50 trend direction. Exits when price returns to Donchian midpoint or trend reverses.
# Designed for ~20-30 trades/year with strict entry conditions to avoid overtrading.
# Works in both bull and bear markets by following the 1d trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period high/low) on 4h
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(19, n):
        high_max[i] = np.max(high[i-19:i+1])
        low_min[i] = np.min(low[i-19:i+1])
    
    # Donchian midpoint for exit
    dc_mid = (high_max + low_min) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(dc_mid[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from 1d EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and bullish trend
            if price > high_max[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and bearish trend
            elif price < low_min[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below Donchian midpoint or trend turns bearish
            if price < dc_mid[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above Donchian midpoint or trend turns bullish
            if price > dc_mid[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0