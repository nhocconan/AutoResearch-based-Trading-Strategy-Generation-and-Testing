#!/usr/bin/env python3
"""
12h_market_regime_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price touches weekly Camarilla S3 level during weekly uptrend (price above weekly SMA50) with volume > 1.5x average; enter short when price touches weekly R3 level during weekly downtrend (price below weekly SMA50) with volume > 1.5x average. Uses weekly trend filter to avoid counter-trend trades. Target: 15-35 trades/year to minimize fee drift while capturing institutional pivot reactions in all market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_market_regime_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly high, low, close for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate Camarilla pivot levels: H-L range
    hl_range = high_1w - low_1w
    close_prev = close_1w  # Using same bar's close for pivot calculation (standard)
    
    # Camarilla levels
    s3 = close_prev - (hl_range * 1.125 / 6)  # Support level 3
    r3 = close_prev + (hl_range * 1.125 / 6)  # Resistance level 3
    
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(vol_ma[i]) or np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price moves below S3 or trend changes to down
            if low[i] < s3_aligned[i] or close[i] < sma_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above R3 or trend changes to up
            if high[i] > r3_aligned[i] or close[i] > sma_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches or penetrates S3 during weekly uptrend
                if (low[i] <= s3_aligned[i] and 
                    close[i] > sma_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or penetrates R3 during weekly downtrend
                elif (high[i] >= r3_aligned[i] and 
                      close[i] < sma_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals