#!/usr/bin/env python3
"""
1d_Weekly_SMA_Crossover_Bullish_Momentum
Hypothesis: Uses weekly SMA crossover (SMA50 crosses SMA200) as primary trend filter,
combined with daily price action for entry timing. Enters long when price pulls back
to weekly SMA50 during bullish trend, exits when momentum weakens. Designed for
low trade frequency (~10-20 trades/year) to minimize fee drag and capture major
trends in both bull and bear markets by following the weekly trend.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly SMAs for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    
    # Align weekly SMAs to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Daily SMA20 for entry timing
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(sma_200_1w_aligned[i]) or 
            np.isnan(sma_20_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        sma_50 = sma_50_1w_aligned[i]
        sma_200 = sma_200_1w_aligned[i]
        sma_20 = sma_20_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Bullish trend: weekly SMA50 > SMA200
        bullish_trend = sma_50 > sma_200
        
        if position == 0:
            # Enter long when: bullish trend + price above weekly SMA50 + 
            # price pulls back to or near daily SMA20 + volume confirmation
            if (bullish_trend and 
                close_val > sma_50 and 
                close_val <= sma_20 * 1.02 and  # Allow 2% above SMA20
                vol_conf):
                signals[i] = size
                position = 1
        elif position == 1:
            # Exit when: trend turns bearish OR price breaks below weekly SMA50
            if not bullish_trend or close_val < sma_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
    
    return signals

name = "1d_Weekly_SMA_Crossover_Bullish_Momentum"
timeframe = "1d"
leverage = 1.0