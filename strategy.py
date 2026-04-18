#!/usr/bin/env python3
"""
6h Daily Close Reversion with Volume Confirmation
Hypothesis: Price tends to revert to the daily close after 6h extremes, especially when accompanied by volume spikes.
Works in both bull and bear markets as it captures mean reversion within the daily context, avoiding trend-following whipsaws.
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
    
    # Get daily data for reference (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily close as mean reversion target
    daily_close = df_1d['close'].values
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Volume spike detection: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Price deviation from daily close: how far price has moved from daily close
    price_deviation = (close - daily_close_aligned) / daily_close_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(daily_close_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        dev = price_deviation[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long when price is significantly below daily close with volume confirmation
            if dev < -0.015 and vol_ok:  # 1.5% below daily close
                signals[i] = 0.25
                position = 1
            # Enter short when price is significantly above daily close with volume confirmation
            elif dev > 0.015 and vol_ok:  # 1.5% above daily close
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to daily close or moves further away
            if dev >= -0.005:  # Within 0.5% of daily close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to daily close or moves further away
            if dev <= 0.005:  # Within 0.5% of daily close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Daily_Close_Reversion_Volume"
timeframe = "6h"
leverage = 1.0