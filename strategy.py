#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly trend filter and volume confirmation
# Weekly trend: price above/below weekly 50 SMA determines long/short bias
# Entry: price breaks Donchian(20) high/low with volume > 1.5x 6-period average
# Exit: price crosses weekly 50 SMA or Donchian middle band
# Target: 15-30 trades/year to minimize fee drag, works in bull/bear via trend filter

name = "6h_1w_donchian_trend_v1"
timeframe = "6h"
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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly 50 SMA (simple moving average)
    weekly_close = df_1w['close'].values
    sma_50_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i >= 49:
            sma_50_1w[i] = np.mean(weekly_close[i-49:i+1])
    
    # Align weekly SMA to 6h timeframe
    sma_50_1w_6h = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Donchian channels (20-period) on 6h
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
            donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Volume confirmation: 6-period average (24h)
    vol_ma_6 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 6:
            vol_sum -= volume[i-6]
        if i >= 5:
            vol_ma_6[i] = vol_sum / 6
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(sma_50_1w_6h[i]) or 
            np.isnan(vol_ma_6[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly 50 SMA OR Donchian middle band
            if close[i] < sma_50_1w_6h[i] or close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly 50 SMA OR Donchian middle band
            if close[i] > sma_50_1w_6h[i] or close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume ratio
            vol_ratio = volume[i] / vol_ma_6[i] if vol_ma_6[i] > 0 else 0
            
            # Enter long: price breaks above Donchian high with volume confirmation AND above weekly SMA
            if (close[i] > donch_high[i] and 
                vol_ratio > 1.5 and 
                close[i] > sma_50_1w_6h[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation AND below weekly SMA
            elif (close[i] < donch_low[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < sma_50_1w_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals