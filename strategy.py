#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Breakout above upper Donchian channel with volume and bullish 12h trend = long.
# Breakdown below lower Donchian channel with volume and bearish 12h trend = short.
# Uses 12h EMA for trend filter to avoid counter-trend trades.
# Target: 20-40 trades per year to minimize fee drag on 4h timeframe.
# Works in both bull and bear markets via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 12h EMA trend filter (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5 x 30-period average
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (50), volume MA (30)
    start_idx = max(20, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume + bullish 12h trend
            if price > upper and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + volume + bearish 12h trend
            elif price < lower and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend turns bearish
            if price < lower or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend turns bullish
            if price > upper or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0