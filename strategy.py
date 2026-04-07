#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v4
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from daily timeframe with volume confirmation. 
Enter long when price breaks above R4 with volume > 1.5x average, short when price breaks below S4 with volume > 1.5x average. 
Exit when price touches opposite pivot level (S4 for long, R4 for short). 
Reduced frequency: increased volume threshold to 2.0x average and added ATR volatility filter to reduce trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v4"
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
    
    # Align to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 20-period average volume for confirmation (lower period for 12h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume average warmup
        # Skip if daily data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (more strict)
        vol_confirm = volume[i] > 2.0 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # Volatility filter: only trade when ATR > 50-period average (avoid low volatility chop)
        atr_avg = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > atr_avg[i] if not np.isnan(atr_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below S4
            if close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above R4
            if close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume confirmation and volatility filter
            long_entry = (close[i] > r4_aligned[i]) and vol_confirm and vol_filter
            # Short entry: price breaks below S4 with volume confirmation and volatility filter
            short_entry = (close[i] < s4_aligned[i]) and vol_confirm and vol_filter
            
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
    
    return signals