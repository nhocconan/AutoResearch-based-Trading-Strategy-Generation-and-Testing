#!/usr/bin/env python3
"""
6h_price_median_reversion_1d_v1
Hypothesis: Price reverts to daily median price with volume confirmation and volatility filter.
Uses daily median (mid-range) as mean reversion target. Works in both bull/bear markets by
fading extremes with volume confirmation and ATR-based volatility filter.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_price_median_reversion_1d_v1"
timeframe = "6h"
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
    
    # Daily data for median price and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily median price (mid-range of previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    daily_median = (prev_high + prev_low) / 2
    
    # Calculate daily ATR(14) for volatility filter
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr.values).rolling(window=14, min_periods=14).mean().values
    
    # Align to 6H timeframe
    daily_median_aligned = align_htf_to_ltf(prices, df_1d, daily_median)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Distance from median in ATR units
    dist_from_median = np.abs(close - daily_median_aligned)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(daily_median_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Volatility filter: only trade when ATR > 50% of its 50-period average
        atr_ma = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_aligned[i] > 0.5 * atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        if position == 1:  # Long position
            # Exit: price crosses above median or volatility drops
            if close[i] >= daily_median_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses below median or volatility drops
            if close[i] <= daily_median_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price below median by 1.5 ATR with volume and vol filter
            if (close[i] < daily_median_aligned[i] - 1.5 * atr_aligned[i] and
                vol_confirm and 
                vol_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price above median by 1.5 ATR with volume and vol filter
            elif (close[i] > daily_median_aligned[i] + 1.5 * atr_aligned[i] and
                  vol_confirm and 
                  vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals