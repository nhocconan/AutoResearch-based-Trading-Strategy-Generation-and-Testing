#!/usr/bin/env python3
"""
4h_Stochastic_Momentum_Trend_Filter
Strategy: 4h Stochastic momentum with 1d trend filter and volume confirmation.
Long: %K > 50 and rising, price above 1d EMA34, volume > 1.8x average
Short: %K < 50 and falling, price below 1d EMA34, volume > 1.8x average
Exit: %K crosses back to 50
Position size: 0.25
Designed to capture momentum moves aligned with daily trend.
Timeframe: 4h
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
    
    # Calculate Stochastic Oscillator %K (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1, denominator)
    
    stoch_raw = (close - lowest_low) / denominator
    stoch_k = pd.Series(stoch_raw).rolling(window=3, min_periods=3).mean().values
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Stochastic momentum: current %K vs previous %K
    stoch_momentum = stoch_k - np.roll(stoch_k, 1)
    stoch_momentum[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(stoch_k[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # Stochastic conditions
        stoch_above_50 = stoch_k[i] > 50
        stoch_below_50 = stoch_k[i] < 50
        stoch_rising = stoch_momentum[i] > 0
        stoch_falling = stoch_momentum[i] < 0
        
        if position == 0:
            # Long: %K > 50 and rising + price above EMA + volume filter
            if stoch_above_50 and stoch_rising and price_above_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: %K < 50 and falling + price below EMA + volume filter
            elif stoch_below_50 and stoch_falling and price_below_ema and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: %K falls back to 50
            if stoch_k[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: %K rises back to 50
            if stoch_k[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Stochastic_Momentum_Trend_Filter"
timeframe = "4h"
leverage = 1.0