#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_v2
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation.
# Long: price touches L3 support with volume spike. Short: price touches H3 resistance with volume spike.
# Exit: opposite pivot level touch or mean reversion to pivot point.
# Uses 12h EMA200 as trend filter to avoid counter-trend trades in strong trends.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = P + 1.1 * Range / 2
    # H3 = P + 1.1 * Range / 4
    # H2 = P + 1.1 * Range / 6
    # H1 = P + 1.1 * Range / 12
    # L1 = P - 1.1 * Range / 12
    # L2 = P - 1.1 * Range / 6
    # L3 = P - 1.1 * Range / 4
    # L4 = P - 1.1 * Range / 2
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    pivot_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    
    h3_1d = pivot_1d + 1.1 * range_1d / 4.0
    l3_1d = pivot_1d - 1.1 * range_1d / 4.0
    
    # Align HTF levels to LTF (wait for completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 12h EMA200 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(ema200_12h_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (mean reversion) or volume divergence
            if high[i] >= h3_1d_aligned[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (mean reversion) or volume divergence
            if low[i] <= l3_1d_aligned[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price touches L3 with volume confirmation and above 12h EMA200
            if (low[i] <= l3_1d_aligned[i] and volume_confirmed and 
                close[i] > ema200_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3 with volume confirmation and below 12h EMA200
            elif (high[i] >= h3_1d_aligned[i] and volume_confirmed and 
                  close[i] < ema200_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals