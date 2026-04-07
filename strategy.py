#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v2
Hypothesis: On 4-hour timeframe, use Camarilla pivot levels from daily timeframe with volume confirmation and stricter entry conditions. 
Enter long when price breaks above R4 with volume > 2.0x average and price > 200 EMA, short when price breaks below S4 with volume > 2.0x average and price < 200 EMA. 
Exit when price touches opposite pivot level (S4 for long, R4 for short). 
Reduced trade frequency by increasing volume threshold and adding trend filter to avoid whipsaws. Designed for low frequency (20-50 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close) / 3
    range_val = d_high - d_low
    
    # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
    r4 = d_close + range_val * 1.1 / 2
    s4 = d_close - range_val * 1.1 / 2
    
    # Align to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 50-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 200 EMA for trend filter
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA and volume average warmup
        # Skip if daily data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 50-period average
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
            # Long entry: price breaks above R4 with volume confirmation AND price > 200 EMA (uptrend)
            long_entry = (close[i] > r4_aligned[i]) and vol_confirm and (close[i] > ema_200[i])
            # Short entry: price breaks below S4 with volume confirmation AND price < 200 EMA (downtrend)
            short_entry = (close[i] < s4_aligned[i]) and vol_confirm and (close[i] < ema_200[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals