#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points (Camarilla R1/S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot and levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily levels to daily timeframe (no interpolation needed for 1d)
    # Since we're using 1d timeframe, we can use the values directly with proper shift
    # For daily timeframe, we need to use previous day's levels to avoid look-ahead
    pivot_shifted = np.roll(pivot, 1)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    pivot_shifted[0] = np.nan
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1w EMA (200 periods), volume MA (20 periods)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(pivot_shifted[i]) or 
            np.isnan(r1_shifted[i]) or np.isnan(s1_shifted[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_200_1w_aligned[i]
        pivot_level = pivot_shifted[i]
        r1_level = r1_shifted[i]
        s1_level = s1_shifted[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1 + bullish trend + volume spike
            if price > r1_level and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 + bearish trend + volume spike
            elif price < s1_level and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to S1 (mean reversion to support) or trend turns bearish
            if price <= s1_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to R1 (mean reversion to resistance) or trend turns bullish
            if price >= r1_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0