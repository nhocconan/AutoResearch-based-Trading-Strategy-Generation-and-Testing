#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: On 12h timeframe, use Camarilla pivot levels from 1d timeframe with volume confirmation.
# Long when price touches or breaks above S3 level with volume > 1.5x average.
# Short when price touches or breaks below R3 level with volume > 1.5x average.
# Exit when price returns to H4 level (mean reversion target) or opposite signal triggers.
# Camarilla levels provide strong intraday support/resistance with high win rate.
# Volume confirmation filters false breakouts. Target: 15-25 trades/year to stay within limits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas:
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    # Pivot = (high + low + close) / 3
    
    # We'll use H3 and L3 as entry/exit levels
    hl_range = high_1d - low_1d
    h3 = close_1d + 1.1 * hl_range / 4
    l3 = close_1d - 1.1 * hl_range / 4
    h4 = close_1d + 1.1 * hl_range / 2  # Exit target for longs
    l4 = close_1d - 1.1 * hl_range / 2  # Exit target for shorts
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: 20-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or \
           np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to H4 level or opposite signal
            if close[i] >= h4_aligned[i] or \
               (close[i] <= l3_aligned[i] and volume[i] > 1.5 * avg_volume[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to L4 level or opposite signal
            if close[i] <= l4_aligned[i] or \
               (close[i] >= h3_aligned[i] and volume[i] > 1.5 * avg_volume[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price touches or breaks above L3 with volume
            if close[i] >= l3_aligned[i] and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or breaks below H3 with volume
            elif close[i] <= h3_aligned[i] and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals