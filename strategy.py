#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Reversal
Hypothesis: Price reversals at Camarilla pivot levels (H3/L3) on 1d timeframe, filtered by 12h trend (EMA34) and volume spikes. Works in bull markets by buying L3 bounces in uptrend, and in bear markets by selling H3 rallies in downtrend. Targets 15-25 trades/year by requiring precise pivot alignment, trend confirmation, and volume > 2x average. Uses Camarilla's mathematically derived support/resistance levels which work well in ranging markets like 2025.
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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H3 = close + (high - low) * 1.1/6
    # L3 = close - (high - low) * 1.1/6
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe (wait for day close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h closes
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe (wait for bar close)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price touches L3 and bounces, with volume, and 12h EMA uptrend
            if (low[i] <= L3_aligned[i] and close[i] > L3_aligned[i] and 
                vol_confirm[i] and close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price touches H3 and rejects, with volume, and 12h EMA downtrend
            elif (high[i] >= H3_aligned[i] and close[i] < H3_aligned[i] and 
                  vol_confirm[i] and close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches opposite H3 level or trend changes
            if (close[i] >= H3_aligned[i] or 
                close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches opposite L3 level or trend changes
            if (close[i] <= L3_aligned[i] or 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_Pivot_Reversal"
timeframe = "12h"
leverage = 1.0