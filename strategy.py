#!/usr/bin/env python3
"""
1h_Equity_Volume_Filter
Hypothesis: Use equity curve momentum (price vs 50-period SMA) combined with volume filter (volume > 1.5x 20-period average) on 1h timeframe. Add session filter (08-20 UTC) to reduce noise. This strategy targets mean-reversion in choppy markets and momentum in trending conditions, with strict entry conditions to limit trades to 15-30/year. The equity filter avoids counter-trend trades, while volume confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 50-period SMA for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for SMA and volume average
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(sma_50[i]) or np.isnan(volume_confirm[i]) or 
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        sma_val = sma_50[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price crosses above SMA50 with volume confirmation
            if close[i] > sma_val and close[i-1] <= sma_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price crosses below SMA50 with volume confirmation
            elif close[i] < sma_val and close[i-1] >= sma_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below SMA50
            if close[i] < sma_val and close[i-1] >= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above SMA50
            if close[i] > sma_val and close[i-1] <= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Equity_Volume_Filter"
timeframe = "1h"
leverage = 1.0