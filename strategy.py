#!/usr/bin/env python3
"""
1d_weekly_sma_breakout_v1
Hypothesis: Weekly SMA crossovers signal long-term trend changes. On daily timeframe, 
enter long when price crosses above weekly SMA20 with momentum confirmation, 
enter short when price crosses below weekly SMA20 with momentum confirmation.
Uses weekly trend filter to avoid counter-trend trades and reduce whipsaw in ranging markets.
Target: 20-40 trades/year on daily timeframe to minimize fee drag.
Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_sma_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for SMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly SMA20 calculation
    close_1w = df_1w['close'].values
    sma_20w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Daily SMA50 for momentum confirmation
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: volume > 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA20 to daily timeframe
    sma_20w_aligned = align_htf_to_ltf(prices, df_1w, sma_20w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(sma_20w_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(sma_50[i])):
            signals[i] = 0.0
            continue
        
        sma_weekly = sma_20w_aligned[i]
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly SMA20 or momentum turns bearish
            if close[i] < sma_weekly or close[i] < sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly SMA20 or momentum turns bullish
            if close[i] > sma_weekly or close[i] > sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price crosses above weekly SMA20 with bullish momentum and volume
            if close[i] > sma_weekly and close[i] > sma_50[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price crosses below weekly SMA20 with bearish momentum and volume
            elif close[i] < sma_weekly and close[i] < sma_50[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals