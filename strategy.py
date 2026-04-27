#!/usr/bin/env python3
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
    
    # Get weekly data for calculations (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Exponential Moving Average (10-period) for trend
    close_1w = df_1w['close'].values
    ema_10_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 10:
        multiplier = 2 / (10 + 1)
        ema_10_1w[9] = np.mean(close_1w[:10])
        for i in range(10, len(close_1w)):
            ema_10_1w[i] = (close_1w[i] * multiplier) + (ema_10_1w[i-1] * (1 - multiplier))
    
    # Calculate weekly Donchian Channel (20-period)
    high_20_1w = np.full(len(close_1w), np.nan)
    low_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        for i in range(19, len(close_1w)):
            high_20_1w[i] = np.max(high[i-19:i+1]) if i < len(high) else np.nan
            low_20_1w[i] = np.min(low[i-19:i+1]) if i < len(low) else np.nan
    
    # Align weekly indicators to daily timeframe
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    # Calculate daily volume average (20-period) for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(10, 20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1w_aligned[i]) or 
            np.isnan(high_20_1w_aligned[i]) or 
            np.isnan(low_20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        if position == 0:
            # Long: Price above weekly EMA10 and breaks above weekly Donchian high with volume
            if price > ema_10_1w_aligned[i] and price > high_20_1w_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA10 and breaks below weekly Donchian low with volume
            elif price < ema_10_1w_aligned[i] and price < low_20_1w_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below weekly EMA10 or volatility spike (potential reversal)
            if price < ema_10_1w_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above weekly EMA10 or volatility spike (potential reversal)
            if price > ema_10_1w_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA10_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0