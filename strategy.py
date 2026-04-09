#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation
# Uses 12h Donchian(20) breakout with volume > 1.3x 20-period average
# Enters only when 1w close > 1w SMA50 (bullish trend filter)
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 15-30 trades/year per symbol to minimize fee drag

name = "12h_1w_donchian_trend_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_high_12h = np.full(len(df_12h), np.nan)
    donch_low_12h = np.full(len(df_12h), np.nan)
    
    for i in range(20, len(df_12h)):
        donch_high_12h[i] = np.max(high_12h[i-20:i])
        donch_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Align 12h Donchian to 12h timeframe (only use completed 12h bars)
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 1w SMA(50) for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = np.full(len(df_1w), np.nan)
    for i in range(50, len(df_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align 1w SMA50 to 12h timeframe (only use completed weekly bars)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume confirmation: 20-period average on 12h
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
    
    for i in range(50, n):  # Start after SMA warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_12h_aligned[i]) or 
            np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low
            if close[i] <= donch_low_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high
            if close[i] >= donch_high_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 12h Donchian high with volume confirmation and bullish trend
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high_12h_aligned[i] and 
                vol_ratio > 1.3 and
                close_1w[-1] > sma_50_1w_aligned[i] if len(close_1w) > 0 else False):  # 1w close > SMA50
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h Donchian low with volume confirmation and bearish trend
            elif (close[i] < donch_low_12h_aligned[i] and 
                  vol_ratio > 1.3 and
                  close_1w[-1] < sma_50_1w_aligned[i] if len(close_1w) > 0 else False):  # 1w close < SMA50
                position = -1
                signals[i] = -0.25
    
    return signals