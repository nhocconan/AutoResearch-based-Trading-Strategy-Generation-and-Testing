#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 22:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA22 trend
    ema_22_4h = pd.Series(close_4h).ewm(span=22, adjust=False, min_periods=22).mean().values
    ema_22_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_22_4h)
    trend_up = close > ema_22_4h_aligned
    trend_down = close < ema_22_4h_aligned
    
    # 1h Camarilla pivot levels (using previous day)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    # Calculate daily pivot from previous day's OHLC
    # We'll use daily data to compute pivot, then apply to intraday
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot and ranges
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels: R1 = close + 1.1*range/12, S1 = close - 1.1*range/12
    r1_1d = close_1d[:-1] + 1.1 * range_1d / 12.0
    s1_1d = close_1d[:-1] - 1.1 * range_1d / 12.0
    
    # Align daily levels to 1h (each daily value lasts 24h)
    r1_1h = align_htf_to_ltf(prices, df_1d[:-1], r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d[:-1], s1_1d)
    
    # Volume spike: current volume > 1.8x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~6 hours to reduce trade frequency
    
    start_idx = max(24, 1)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_22_4h_aligned[i]) or 
            np.isnan(r1_1h[i]) or 
            np.isnan(s1_1h[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above Camarilla R1 with volume spike in 4h uptrend
            if (close[i] > r1_1h[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below Camarilla S1 with volume spike in 4h downtrend
            elif (close[i] < s1_1h[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below Camarilla S1 or 4h trend changes to down
            if close[i] < s1_1h[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above Camarilla R1 or 4h trend changes to up
            if close[i] > r1_1h[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout captures institutional breakouts in both bull and bear markets.
# Long when price breaks above Camarilla R1 with volume spike and 4h uptrend.
# Short when price breaks below Camarilla S1 with volume spike and 4h downtrend.
# Works in bull markets (sustained uptrend with breakouts above R1) and bear markets (sustained downtrend with breakdowns below S1).
# Volume spike confirms institutional participation. 4h trend filter ensures we trade with higher timeframe momentum.
# 1h timeframe provides precise entry timing while 4h/1d filters reduce noise.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag. Discrete size 0.20 minimizes churn.