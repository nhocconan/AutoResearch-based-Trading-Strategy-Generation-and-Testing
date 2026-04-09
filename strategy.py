#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and weekly trend filter
# Works in bull/bear by using breakouts with volume and weekly trend alignment
# Target: 20-40 trades/year to avoid fee drag, focusing on high-probability breakouts

name = "1d_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(df_1w), np.nan)
    multiplier = 2 / (20 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_20_1w[i] = close_1w[i]
        elif np.isnan(ema_20_1w[i-1]):
            ema_20_1w[i] = close_1w[i]
        else:
            ema_20_1w[i] = close_1w[i] * multiplier + ema_20_1w[i-1] * (1 - multiplier)
    
    # Align weekly EMA to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA20
            if close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA20
            if close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian upper with volume confirmation AND price above weekly EMA20
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > upper[i] and 
                vol_ratio > 1.5 and 
                close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower with volume confirmation AND price below weekly EMA20
            elif (close[i] < lower[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals