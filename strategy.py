#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w EMA trend filter and volume confirmation
# Uses 1w EMA(50) for trend direction to avoid counter-trend trades
# Enters on breakout of 20-period 1d Donchian channels with volume > 2x 20-day average
# Exits on opposite Donchian channel touch or close below/above EMA
# Position size 0.25 to manage drawdown
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag
# Works in bull/bear: trend filter ensures we trade with higher timeframe momentum

name = "1d_1w_donchian_ema_vol_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(df_1w), np.nan)
    if len(close_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Align 1w EMA to 1d timeframe
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-day average
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
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian or closes below EMA
            if low[i] <= donchian_low[i] or close[i] < ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian or closes above EMA
            if high[i] >= donchian_high[i] or close[i] > ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper Donchian with volume confirmation and uptrend
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (high[i] > donchian_high[i] and 
                close[i] > ema_50_1d[i] and 
                vol_ratio > 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower Donchian with volume confirmation and downtrend
            elif (low[i] < donchian_low[i] and 
                  close[i] < ema_50_1d[i] and 
                  vol_ratio > 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals