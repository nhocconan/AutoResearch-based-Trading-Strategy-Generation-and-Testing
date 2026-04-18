#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pullback_Reversal_V1
Hypothesis: Price often pulls back to Camarilla H3/L3 levels before continuing the 12h trend. 
Long when: 12h EMA34 up, price pulls back to H3 and bounces, volume confirms.
Short when: 12h EMA34 down, price pulls back to L3 and rejects, volume confirms.
Uses tight entries (pullback to H3/L3) to limit trades to ~25-35/year. Works in bull by buying dips in uptrend, 
in bear by selling rallies in downtrend. Volume filter avoids false breakouts.
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
    
    # Get 12h data for EMA trend filter and Camarilla calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h Camarilla H3 and L3
    range_12h = high_12h - low_12h
    h3_12h = close_12h + range_12h * 1.1 / 6
    l3_12h = close_12h - range_12h * 1.1 / 6
    
    # Align H3/L3 to 4h timeframe
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(h3_12h_aligned[i]) or 
            np.isnan(l3_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to H3 and bounces in 12h uptrend
            if (ema_12h_aligned[i] > ema_12h_aligned[i-1] and  # EMA rising
                low[i] <= h3_12h_aligned[i] * 1.002 and  # touched H3 (0.2% buffer)
                close[i] > h3_12h_aligned[i] * 1.001 and  # closed above H3
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to L3 and rejects in 12h downtrend
            elif (ema_12h_aligned[i] < ema_12h_aligned[i-1] and  # EMA falling
                  high[i] >= l3_12h_aligned[i] * 0.998 and  # touched L3
                  close[i] < l3_12h_aligned[i] * 0.999 and  # closed below L3
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: EMA turns down or price breaks below L3 (failed hold)
            if (ema_12h_aligned[i] < ema_12h_aligned[i-1] or 
                close[i] < l3_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA turns up or price breaks above H3 (failed hold)
            if (ema_12h_aligned[i] > ema_12h_aligned[i-1] or 
                close[i] > h3_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_Pullback_Reversal_V1"
timeframe = "4h"
leverage = 1.0