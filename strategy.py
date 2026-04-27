#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with volume confirmation and 12h trend filter.
# Long when price breaks above 20-period high with volume > 1.5x average and 12h close > EMA34.
# Short when price breaks below 20-period low with volume > 1.5x average and 12h close < EMA34.
# Exit when price returns to 20-period midline or trend reverses.
# Uses discrete position sizing (0.25) to limit turnover and fees.
# Designed for ~20-30 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Donchian channels (20-period high/low)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(19, n):
        high_max[i] = np.max(high[i-19:i+1])
        low_min[i] = np.min(low[i-19:i+1])
    
    # Midline for exit
    mid_line = (high_max + low_min) / 2.0
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 30-period volume MA and 20-period Donchian
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(mid_line[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from 12h EMA34
        bullish_trend = price > ema34_aligned[i]
        bearish_trend = price < ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with volume and bullish trend
            if price > high_max[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below 20-period low with volume and bearish trend
            elif price < low_min[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below midline or trend turns bearish
            if price < mid_line[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above midline or trend turns bullish
            if price > mid_line[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_Volume_12hTrend"
timeframe = "6h"
leverage = 1.0