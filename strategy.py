#!/usr/bin/env python3
"""
4h Pivot Range Breakout with Volume Confirmation and 1d EMA Filter
Hypothesis: Daily pivot ranges act as key support/resistance. Breaking above the daily pivot range 
with volume confirmation and 1d EMA trend filter captures momentum breakouts. Works in bull markets 
via upward breaks above resistance and in bear markets via downward breaks below support. 
Low trade frequency due to requirement of breaking the full pivot range (not just single levels).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot range and EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot range: (H+L+C)/3 ± (H-L)/2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    range_val = (df_1d['high'] - df_1d['low']).values
    pivot_high = pivot + range_val / 2  # Resistance level
    pivot_low = pivot - range_val / 2   # Support level
    
    # Align pivot levels to 4h timeframe
    pivot_high_aligned = align_htf_to_ltf(prices, df_1d, pivot_high)
    pivot_low_aligned = align_htf_to_ltf(prices, df_1d, pivot_low)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(pivot_high_aligned[i]) or np.isnan(pivot_low_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_1d_aligned[i]
        vol_ok = vol_confirm[i]
        ph = pivot_high_aligned[i]
        pl = pivot_low_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above daily pivot high (resistance) with volume + uptrend
            if vol_ok and close[i] > ph and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily pivot low (support) with volume + downtrend
            elif vol_ok and close[i] < pl and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot low or trend turns down
            if close[i] < pl or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot high or trend turns up
            if close[i] > ph or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_Range_Breakout_Volume_EMA"
timeframe = "4h"
leverage = 1.0