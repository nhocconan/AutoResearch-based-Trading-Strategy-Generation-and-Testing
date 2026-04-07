#!/usr/bin/env python3
"""
4h_ema21_trend_breakout_volume_v1
Hypothesis: On 4h timeframe, enter long when price breaks above EMA21 with volume > 1.5x average during bullish regime (price > 1d SMA50), enter short when price breaks below EMA21 with volume > 1.5x average during bearish regime (price < 1d SMA50). Uses 1d SMA50 for trend filter to avoid counter-trend trades. Target: 25-35 trades/year to minimize fee drag while capturing trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema21_trend_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA21
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
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
    
    for i in range(21, n):  # Start after EMA21 warmup
        # Skip if data not available
        if (np.isnan(ema21[i]) or np.isnan(vol_ma[i]) or np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA21 or trend changes to bearish
            if close[i] < ema21[i] or close[i] < sma_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA21 or trend changes to bullish
            if close[i] > ema21[i] or close[i] > sma_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above EMA21 in bullish regime (price > 1d SMA50)
                if close[i] > ema21[i] and close[i] > sma_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below EMA21 in bearish regime (price < 1d SMA50)
                elif close[i] < ema21[i] and close[i] < sma_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals