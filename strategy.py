#!/usr/bin/env python3
# 6h_Stochastic_Trend_Scalper
# Hypothesis: Combines 6h stochastic oscillator (14,3,3) with 1d EMA trend filter.
# In bullish trend (price > 1d EMA50), buy when stochastic crosses above 20 from below.
# In bearish trend (price < 1d EMA50), sell when stochastic crosses below 80 from above.
# Uses volume confirmation (volume > 1.5x 20-period average) to filter false signals.
# Designed for 15-35 trades/year on 6h timeframe with controlled risk exposure.

name = "6h_Stochastic_Trend_Scalper"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Stochastic Oscillator (14,3,3) on 6h data
    lookback = 14
    lowest_low = np.full_like(low, np.nan)
    highest_high = np.full_like(high, np.nan)
    
    for i in range(lookback - 1, n):
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
    
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    stoch_k = np.full_like(close, np.nan)
    valid_range = (highest_high - lowest_low) != 0
    stoch_k[valid_range] = (close[valid_range] - lowest_low[valid_range]) / \
                           (highest_high[valid_range] - lowest_low[valid_range]) * 100
    
    # %D = 3-period SMA of %K
    stoch_d = np.full_like(close, np.nan)
    if n >= 3:
        for i in range(2, n):
            if not np.isnan(stoch_k[i-2]) and not np.isnan(stoch_k[i-1]) and not np.isnan(stoch_k[i]):
                stoch_d[i] = (stoch_k[i-2] + stoch_k[i-1] + stoch_k[i]) / 3
    
    # Volume filter: 6h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(stoch_d[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bullish trend + stochastic crosses above 20 + volume confirmation
            if (close[i] > ema_50_1d_aligned[i] and 
                stoch_d[i] > 20 and stoch_d[i-1] <= 20 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish trend + stochastic crosses below 80 + volume confirmation
            elif (close[i] < ema_50_1d_aligned[i] and 
                  stoch_d[i] < 80 and stoch_d[i-1] >= 80 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns bearish OR stochastic crosses below 80
            if close[i] < ema_50_1d_aligned[i] or (stoch_d[i] < 80 and stoch_d[i-1] >= 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns bullish OR stochastic crosses above 20
            if close[i] > ema_50_1d_aligned[i] or (stoch_d[i] > 20 and stoch_d[i-1] <= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals