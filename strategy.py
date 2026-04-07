#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_volume_v1
Hypothesis: On 12-hour timeframe, use weekly Camarilla pivot levels with volume confirmation and 1-day EMA200 trend filter. 
Enter long when price touches S1 in uptrend, short when price touches R1 in downtrend. 
Exit when price crosses the pivot (H5/L5) or trend reverses. 
Designed for low frequency (10-20 trades/year) to avoid fee drag while capturing reversals at key levels in both bull and bear markets.
Uses volume spike (1.5x average) to confirm institutional interest at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels: H4, H3, H2, H1, L1, L2, L3, L4"""
    # Pivot point
    pivot = (high + low + close) / 3.0
    # Range
    range_val = high - low
    
    # Camarilla levels
    h4 = close + range_val * 1.1 / 2
    h3 = close + range_val * 1.1 / 4
    h2 = close + range_val * 1.1 / 6
    h1 = close + range_val * 1.1 / 12
    l1 = close - range_val * 1.1 / 12
    l2 = close - range_val * 1.1 / 6
    l3 = close - range_val * 1.1 / 4
    l4 = close - range_val * 1.1 / 2
    
    return pivot, h1, h2, h3, h4, l1, l2, l3, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    w_volume = df_1w['volume'].values
    
    # Calculate weekly Camarilla levels
    camarilla_data = np.array([calculate_camarilla(w_high[i], w_low[i], w_close[i]) 
                               for i in range(len(w_close))])
    # Columns: pivot, h1, h2, h3, h4, l1, l2, l3, l4
    w_pivot = camarilla_data[:, 0]
    w_h1 = camarilla_data[:, 1]
    w_h2 = camarilla_data[:, 2]
    w_h3 = camarilla_data[:, 3]
    w_h4 = camarilla_data[:, 4]
    w_l1 = camarilla_data[:, 5]
    w_l2 = camarilla_data[:, 6]
    w_l3 = camarilla_data[:, 7]
    w_l4 = camarilla_data[:, 8]
    
    # Align to 12h timeframe
    w_pivot_aligned = align_htf_to_ltf(prices, df_1w, w_pivot)
    w_h1_aligned = align_htf_to_ltf(prices, df_1w, w_h1)
    w_l1_aligned = align_htf_to_ltf(prices, df_1w, w_l1)
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ma_aligned = vol_ma.values  # already same length as prices
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if Camarilla levels not available
        if np.isnan(w_pivot_aligned[i]) or np.isnan(w_h1_aligned[i]) or np.isnan(w_l1_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if daily EMA200 not available
        if np.isnan(d_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses above pivot (take profit)
            if close[i] > w_pivot_aligned[i]:
                exit_long = True
            # Exit when trend turns down
            elif not uptrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses below pivot (take profit)
            if close[i] < w_pivot_aligned[i]:
                exit_short = True
            # Exit when trend turns up
            elif not downtrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or goes below L1 in uptrend with volume spike
            long_entry = (close[i] <= w_l1_aligned[i]) and uptrend and vol_spike[i]
            # Short entry: price touches or goes above H1 in downtrend with volume spike
            short_entry = (close[i] >= w_h1_aligned[i]) and downtrend and vol_spike[i]
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals