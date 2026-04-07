#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price touches weekly Camarilla L3 with price above weekly SMA50 and volume > 1.8x average, enter short when price touches weekly H3 with price below weekly SMA50 and volume > 1.8x average. Uses weekly trend filter and volume confirmation to capture mean-reversion bounces within the weekly trend. Target: 20-30 trades/year to minimize fee drag while capturing institutional reversal points.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Calculate weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly high, low, close for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.8x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.8)
        
        if position == 1:  # Long position
            # Exit: price moves above weekly SMA50 or touches H3 (take profit)
            if close[i] >= sma_50_1w_aligned[i] or high[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below weekly SMA50 or touches L3 (take profit)
            if close[i] <= sma_50_1w_aligned[i] or low[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches L3 in uptrend (close above weekly SMA50)
                if (low[i] <= camarilla_l3_aligned[i] * 1.001 and  # Allow small tolerance
                    close[i] > sma_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H3 in downtrend (close below weekly SMA50)
                elif (high[i] >= camarilla_h3_aligned[i] * 0.999 and  # Allow small tolerance
                      close[i] < sma_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals