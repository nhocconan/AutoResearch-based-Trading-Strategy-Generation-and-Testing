#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Uses 20-period Donchian channels for breakout signals, filtered by 12h EMA50 trend
# and volume > 1.5x 20-period average. Designed for low-frequency, high-conviction
# trades with strict entry conditions to minimize fee drag. Works in both bull and
# bear markets by following the higher timeframe trend.

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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 20-period Donchian channels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(19, n):
        high_20[i] = np.max(high[i-19:i+1])
        low_20[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period Donchian and volume MA
    start_idx = 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Breakout conditions
        upper_breakout = price > high_20[i]
        lower_breakout = price < low_20[i]
        
        # Trend filter from 12h EMA50
        bullish_trend = ema50_12h_aligned[i] > ema50_12h_aligned[i-1] if i > 0 else False
        bearish_trend = ema50_12h_aligned[i] < ema50_12h_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: upward breakout with volume and bullish trend
            if upper_breakout and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: downward breakout with volume and bearish trend
            elif lower_breakout and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retracement to midpoint or trend change
            midpoint = (high_20[i] + low_20[i]) / 2
            if price < midpoint or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price retracement to midpoint or trend change
            midpoint = (high_20[i] + low_20[i]) / 2
            if price > midpoint or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0