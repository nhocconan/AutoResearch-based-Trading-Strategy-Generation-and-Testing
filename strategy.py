#!/usr/bin/env python3
"""
6h_Stochastic_Bollinger_Bands_Squeeze_Breakout
Hypothesis: Bollinger Bands squeeze (low volatility) followed by breakout with Stochastic momentum confirmation works across market regimes. 
- Bollinger Bands width < 20th percentile indicates volatility contraction (squeeze)
- Breakout occurs when price closes outside Bollinger Bands
- Stochastic oscillator > 80 confirms bullish momentum, < 20 confirms bearish momentum
- Works in both bull and breakout markets by capturing volatility expansion after consolidation
- Uses 1d timeframe for Bollinger Bands and Stochastic to avoid noise, 6s for execution
Target: 15-25 trades/year by requiring squeeze + breakout + momentum confluence
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
    
    # Get daily data for Bollinger Bands and Stochastic
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    
    sma = np.full_like(close_1d, np.nan)
    bb_upper = np.full_like(close_1d, np.nan)
    bb_lower = np.full_like(close_1d, np.nan)
    bb_width = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= bb_period:
        for i in range(bb_period-1, len(close_1d)):
            sma[i] = np.mean(close_1d[i-bb_period+1:i+1])
            std = np.std(close_1d[i-bb_period+1:i+1])
            bb_upper[i] = sma[i] + bb_std * std
            bb_lower[i] = sma[i] - bb_std * std
            bb_width[i] = (bb_upper[i] - bb_lower[i]) / sma[i] if sma[i] != 0 else np.nan
    
    # Calculate Stochastic Oscillator (14, 3, 3)
    stoch_k = np.full_like(close_1d, np.nan)
    stoch_d = np.full_like(close_1d, np.nan)
    
    stoch_period = 14
    k_smooth = 3
    d_smooth = 3
    
    if len(close_1d) >= stoch_period:
        for i in range(stoch_period-1, len(close_1d)):
            highest_high = np.max(high_1d[i-stoch_period+1:i+1])
            lowest_low = np.min(low_1d[i-stoch_period+1:i+1])
            if highest_high != lowest_low:
                stoch_k[i] = 100 * (close_1d[i] - lowest_low) / (highest_high - lowest_low)
            else:
                stoch_k[i] = 50.0
        
        # Smooth %K to get slow %K
        if len(close_1d) >= stoch_period + k_smooth - 1:
            for i in range(stoch_period + k_smooth - 2, len(close_1d)):
                stoch_k[i] = np.mean(stoch_k[i-k_smooth+1:i+1])
        
        # Smooth slow %K to get slow %D
        if len(close_1d) >= stoch_period + k_smooth + d_smooth - 2:
            for i in range(stoch_period + k_smooth + d_smooth - 2, len(close_1d)):
                stoch_d[i] = np.mean(stoch_k[i-d_smooth+1:i+1])
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 20
    
    if len(bb_width) >= lookback:
        for i in range(lookback-1, len(bb_width)):
            if not np.isnan(bb_width[i]):
                valid_widths = bb_width[i-lookback+1:i+1]
                valid_widths = valid_widths[~np.isnan(valid_widths)]
                if len(valid_widths) > 0:
                    bb_width_percentile[i] = (np.sum(valid_widths <= bb_width[i]) / len(valid_widths)) * 100
    
    # Align all 1d data to 6h timeframe
    bb_width_percentile_6h = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    bb_upper_6h = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_6h = align_htf_to_ltf(prices, df_1d, bb_lower)
    stoch_k_6h = align_htf_to_ltf(prices, df_1d, stoch_k)
    stoch_d_6h = align_htf_to_ltf(prices, df_1d, stoch_d)
    
    signals = np.zeros(n)
    
    start_idx = max(bb_period, stoch_period + k_smooth + d_smooth, lookback) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_percentile_6h[i]) or np.isnan(bb_upper_6h[i]) or 
            np.isnan(bb_lower_6h[i]) or np.isnan(stoch_k_6h[i]) or np.isnan(stoch_d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band squeeze condition: width < 20th percentile
        squeeze = bb_width_percentile_6h[i] < 20
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper_6h[i]
        breakout_down = close[i] < bb_lower_6h[i]
        
        # Stochastic momentum confirmation
        stoch_bullish = stoch_k_6h[i] > 50 and stoch_d_6h[i] > 50 and stoch_k_6h[i] > stoch_d_6h[i]
        stoch_bearish = stoch_k_6h[i] < 50 and stoch_d_6h[i] < 50 and stoch_k_6h[i] < stoch_d_6h[i]
        
        # Entry signals
        if squeeze and breakout_up and stoch_bullish:
            signals[i] = 0.25
        elif squeeze and breakout_down and stoch_bearish:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Stochastic_Bollinger_Bands_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0