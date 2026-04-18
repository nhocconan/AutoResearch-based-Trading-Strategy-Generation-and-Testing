#!/usr/bin/env python3
"""
6h_1D_Momentum_Confluence
Hypothesis: Use 1D EMA(34) as primary trend filter, with 6h EMA(13) cross above/below 1D EMA(34) for entry, and volume > 1.5x 20-period average for confirmation. This combines trend-following (EMA34) with momentum (EMA13 cross) and volume confirmation to reduce false signals. Works in bull markets by taking longs when 6h momentum aligns with 1D uptrend, and in bear markets by taking shorts when momentum aligns with 1D downtrend. Targets 15-25 trades/year by requiring EMA cross alignment, volume confirmation, and avoiding entries during low momentum periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for EMA34 (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1D
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h timeframe (wait for bar close)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA13 on 6s (LTF)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: EMA13 crosses above EMA34_1d, with volume confirmation
            if (ema13[i] > ema34_1d_aligned[i] and 
                ema13[i-1] <= ema34_1d_aligned[i-1] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: EMA13 crosses below EMA34_1d, with volume confirmation
            elif (ema13[i] < ema34_1d_aligned[i] and 
                  ema13[i-1] >= ema34_1d_aligned[i-1] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: EMA13 crosses below EMA34_1d (trend change)
            if ema13[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA13 crosses above EMA34_1d (trend change)
            if ema13[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_Momentum_Confluence"
timeframe = "6h"
leverage = 1.0