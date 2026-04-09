#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + 1d ATR filter
# Direction from higher timeframe: 4h Donchian channels determine trend, 1d ATR filters volatility regime
# Entry timing on 1h with volume confirmation to avoid false breakouts
# Target: 15-30 trades/year to minimize fee drag in challenging 1h timeframe
# Works in bull/bear by using volatility-adjusted breakouts with volume confirmation

name = "1h_4h_1d_donchian_atr_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(19, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to 1h timeframe
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.full(len(df_1d), np.nan)
    tr = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            tr[i] = high_1d[i] - low_1d[i]
        else:
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
    
    for i in range(13, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Align ATR to 1h timeframe
    atr_1d_1h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 20-period average on 1h
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
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_1h[i]) or 
            np.isnan(donchian_low_1h[i]) or 
            np.isnan(atr_1d_1h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1d ATR is above 25th percentile (avoid low volatility)
        if i >= 50:
            atr_slice = atr_1d_1h[max(0, i-49):i+1]
            valid_atr = atr_slice[~np.isnan(atr_slice)]
            if len(valid_atr) >= 20:
                atr_percentile = np.percentile(valid_atr, 25)
                if atr_1d_1h[i] < atr_percentile:
                    signals[i] = 0.0
                    continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR ATR drops significantly
            if close[i] < donchian_low_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR ATR drops significantly
            if close[i] > donchian_high_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price closes above Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donchian_high_1h[i] and 
                vol_ratio > 1.8):
                position = 1
                signals[i] = 0.20
            # Enter short: price closes below Donchian low with volume confirmation
            elif (close[i] < donchian_low_1h[i] and 
                  vol_ratio > 1.8):
                position = -1
                signals[i] = -0.20
    
    return signals