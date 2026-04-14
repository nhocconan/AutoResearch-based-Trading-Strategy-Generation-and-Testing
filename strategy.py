#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and range analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.mean(tr[i-13:i+1])
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily range percentage (high-low)/close
    daily_range_pct = (high_1d - low_1d) / close_1d
    daily_range_pct_aligned = align_htf_to_ltf(prices, df_1d, daily_range_pct)
    
    # Calculate 4-period SMA of daily range for smoothing
    range_sma = pd.Series(daily_range_pct).rolling(window=4, min_periods=4).mean().values
    range_sma_aligned = align_htf_to_ltf(prices, df_1d, range_sma)
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 20-period SMA on 4h for trend
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    # Start after enough data for calculations
    start = max(20, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(daily_range_pct_aligned[i]) or
            np.isnan(range_sma_aligned[i]) or np.isnan(sma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: low volatility (range contraction) + price above SMA20
            if (daily_range_pct_aligned[i] < range_sma_aligned[i] * 0.8 and  # Contraction
                price > sma_20_4h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: low volatility + price below SMA20
            elif (daily_range_pct_aligned[i] < range_sma_aligned[i] * 0.8 and  # Contraction
                  price < sma_20_4h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: volatility expansion or price crosses SMA20
            if (daily_range_pct_aligned[i] > range_sma_aligned[i] * 1.2 or  # Expansion
                price < sma_20_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: volatility expansion or price crosses SMA20
            if (daily_range_pct_aligned[i] > range_sma_aligned[i] * 1.2 or  # Expansion
                price > sma_20_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RangeContraction_SMA20"
timeframe = "4h"
leverage = 1.0