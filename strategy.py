#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use Camarilla pivot levels from daily chart for mean reversion entries, filtered by 1d EMA trend and volume confirmation. 
In range-bound markets, price tends to revert to mean from L3/H3 levels. In trending markets, EMA filter ensures we only take trades in trend direction.
Volume confirms genuine pivot touches. Target: 20-40 trades/year (~80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use L3 and H3 for mean reversion entries
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate pivot levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pp + (range_hl * 1.1 / 4)  # High 3
    l3 = pp - (range_hl * 1.1 / 4)  # Low 3
    h4 = pp + (range_hl * 1.1 / 2)  # High 4 (stop level)
    l4 = pp - (range_hl * 1.1 / 2)  # Low 4 (stop level)
    
    # Align 1d levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (take profit) or breaks below L4 (stop) or trend changes
            if (close[i] >= h3_aligned[i] or close[i] <= l4_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches L3 (take profit) or breaks above H4 (stop) or trend changes
            if (close[i] <= l3_aligned[i] or close[i] >= h4_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches L3 from below, with volume and above EMA50 (bullish alignment)
            if (close[i] <= l3_aligned[i] * 1.005 and close[i] > l3_aligned[i] * 0.995 and  # Near L3
                vol_confirm and close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3 from above, with volume and below EMA50 (bearish alignment)
            elif (close[i] >= h3_aligned[i] * 0.995 and close[i] < h3_aligned[i] * 1.005 and  # Near H3
                  vol_confirm and close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals