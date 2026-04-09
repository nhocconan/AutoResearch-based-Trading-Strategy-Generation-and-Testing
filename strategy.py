#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly EMA trend filter and volume confirmation
# Works in bull/bear by using weekly EMA to filter breakout direction, reducing false signals
# Target: 20-40 trades/year to avoid fee drag, focusing on high-probability breakouts

name = "1d_1w_donchian_ema_volume_v1"
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
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        multiplier = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * multiplier) + (ema_20_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian channel (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
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
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR weekly EMA turns down
            if close[i] < donch_low[i] or ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR weekly EMA turns up
            if close[i] > donch_high[i] or ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume confirmation AND price above weekly EMA
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high[i] and 
                vol_ratio > 2.0 and 
                close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation AND price below weekly EMA
            elif (close[i] < donch_low[i] and 
                  vol_ratio > 2.0 and 
                  close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals