#!/usr/bin/env python3
# 1d_camarilla_pivot_1w_trend_volume_v1
# Hypothesis: Camarilla pivot levels on 1d with volume confirmation and weekly trend filter
# captures mean-reversion bounces off strong support/resistance in both bull and bear markets.
# Weekly trend filter (EMA20) ensures we only take long in uptrend and short in downtrend.
# Volume spike confirms institutional interest at pivot levels.
# Target: 50-80 total trades over 4 years (~12-20/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays"""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    # L4 = close - range * 1.1/2
    # L3 = close - range * 1.1/4
    # L2 = close - range * 1.1/6
    # L1 = close - range * 1.1/12
    # H1 = close + range * 1.1/12
    # H2 = close + range * 1.1/6
    # H3 = close + range * 1.1/4
    # H4 = close + range * 1.1/2
    
    l4 = close - (range_val * 1.1 / 2)
    l3 = close - (range_val * 1.1 / 4)
    l2 = close - (range_val * 1.1 / 6)
    l1 = close - (range_val * 1.1 / 12)
    h1 = close + (range_val * 1.1 / 12)
    h2 = close + (range_val * 1.1 / 6)
    h3 = close + (range_val * 1.1 / 4)
    h4 = close + (range_val * 1.1 / 2)
    
    return l4, l3, l2, l1, h1, h2, h3, h4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla levels for each day
    l4, l3, l2, l1, h1, h2, h3, h4 = calculate_camarilla(high, low, close)
    
    # Volume filter: today's volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(l1[i]) or np.isnan(h1[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (strong resistance) or breaks below L1 (support broken)
            if (close[i] >= h3[i]) or (close[i] <= l1[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (strong support) or breaks above H1 (resistance broken)
            if (close[i] <= l3[i]) or (close[i] >= h1[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only take long in weekly uptrend, short in weekly downtrend
            weekly_uptrend = close[i] > ema_20_1w_aligned[i]
            weekly_downtrend = close[i] < ema_20_1w_aligned[i]
            
            # Long entry: price touches L3 (strong support) with volume in uptrend
            if weekly_uptrend and volume_filter[i] and (low[i] <= l3[i]) and (close[i] > l3[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3 (strong resistance) with volume in downtrend
            elif weekly_downtrend and volume_filter[i] and (high[i] >= h3[i]) and (close[i] < h3[i]):
                position = -1
                signals[i] = -0.25
    
    return signals