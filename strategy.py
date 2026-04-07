#!/usr/bin/env python3
"""
4h_bollinger_bounce_1d_trend_volume_v1
Hypothesis: On 4h timeframe, enter long when price touches lower Bollinger Band during uptrend (price above 1d SMA50) with volume > 1.5x average, enter short when price touches upper Bollinger Band during downtrend (price below 1d SMA50) with volume > 1.5x average. Uses 1d SMA50 trend filter to avoid counter-trend trades. Bollinger Bands provide mean-reversion signals within trending markets, targeting 20-40 trades/year to minimize fee drag while capturing bounces in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_bounce_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d SMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(bb_period, n):  # Start after Bollinger Bands warmup
        # Skip if data not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price moves back to middle band or trend changes
            if close[i] >= sma[i] or close[i] < sma_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back to middle band or trend changes
            if close[i] <= sma[i] or close[i] > sma_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches lower Bollinger Band in uptrend (price above 1d SMA50)
                if (low[i] <= lower_band[i] and 
                    close[i] > sma_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches upper Bollinger Band in downtrend (price below 1d SMA50)
                elif (high[i] >= upper_band[i] and 
                      close[i] < sma_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals