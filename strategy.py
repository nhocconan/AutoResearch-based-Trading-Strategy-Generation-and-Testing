#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_ema_volume_v1
Hypothesis: On 12-hour timeframe, use weekly (1w) Camarilla pivot levels with EMA trend filter and volume confirmation. 
Enter long when price breaks above R4 with volume > 2.0x average and price > 50 EMA, short when price breaks below S4 with volume > 2.0x average and price < 50 EMA. 
Exit when price touches opposite pivot level (S4 for long, R4 for short). 
Designed for low frequency (12-37 trades/year) to minimize fee drag while capturing multi-week trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    pivot = (w_high + w_low + w_close) / 3
    range_val = w_high - w_low
    
    # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
    r4 = w_close + range_val * 1.1 / 2
    s4 = w_close - range_val * 1.1 / 2
    
    # Align to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 50 EMA for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA and volume average warmup
        # Skip if weekly data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below S4
            if close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above R4
            if close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume confirmation AND price > 50 EMA (uptrend)
            long_entry = (close[i] > r4_aligned[i]) and vol_confirm and (close[i] > ema_50[i])
            # Short entry: price breaks below S4 with volume confirmation AND price < 50 EMA (downtrend)
            short_entry = (close[i] < s4_aligned[i]) and vol_confirm and (close[i] < ema_50[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals