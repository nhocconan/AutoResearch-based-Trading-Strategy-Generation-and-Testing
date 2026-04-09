#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels (L3/L4 for shorts, H3/H4 for longs) with volume confirmation (>1.5x 20-period average). Enters long when price touches or breaks above H3/H4 with volume confirmation; short when price touches or breaks below L3/L4 with volume confirmation. Uses discrete position sizing (0.25) and exits on opposite pivot level touch. Designed for low turnover (target: 12-37 trades/year) by requiring both price level touch and volume spike, working in ranging markets where price respects pivot levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical_price = (high + low + close) / 3
    range_ = high - low
    h4 = close + range_ * 1.1 / 2
    h3 = close + range_ * 1.1 / 4
    l3 = close - range_ * 1.1 / 4
    l4 = close - range_ * 1.1 / 2
    return h3, h4, l3, l4

name = "12h_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    h1 = df_1d['high'].values
    l1 = df_1d['low'].values
    c1 = df_1d['close'].values
    
    h3_1d, h4_1d, l3_1d, l4_1d = calculate_camarilla_pivot(h1, l1, c1)
    
    # Align HTF levels to LTF
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(h3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks below L3 (mean reversion target)
            if close[i] <= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks above H3 (mean reversion target)
            if close[i] >= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation
            if volume_confirmed:
                # Long: price breaks above H4 (bullish breakout)
                if close[i] > h4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L4 (bearish breakout)
                elif close[i] < l4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals