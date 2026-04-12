#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_volume_regime_v1
# Uses weekly Camarilla pivot levels (H3/L3) with volume confirmation and 1d momentum filter.
# In bull markets, buys breakouts above weekly H3 with volume and positive 1d momentum.
# In bear markets, shorts breakdowns below weekly L3 with volume and negative 1d momentum.
# Weekly timeframe reduces noise and false signals; volume confirms institutional interest.
# Target: 15-30 trades/year per symbol for low friction and high edge.

name = "1d_1w_camarilla_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 1d timeframe (already delayed by 1 week due to shift)
    h3_level = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # 1d momentum filter: positive/negative momentum over 3 days
    mom = close - np.roll(close, 3)
    mom = np.where(np.arange(len(close)) < 3, 0, mom)  # pad first 3 values
    mom_filter = mom > 0  # for long
    mom_filter_short = mom < 0  # for short
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly H3 with volume and positive momentum
        if close[i] > h3_level[i] and mom_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly L3 with volume and negative momentum
        elif close[i] < l3_level[i] and mom_filter_short[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h3_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals