#!/usr/bin/env python3
# 12h_camarilla_pivot_1w_trend_volume_v1
# Hypothesis: Use daily Camarilla pivot levels on 12h timeframe with volume and weekly trend confirmation.
# Go long when price closes above L4 (support) in weekly uptrend, short when closes below H4 (resistance) in weekly downtrend.
# Camarilla levels provide statistically significant support/resistance with built-in breakout/reversal logic.
# Weekly trend filter ensures alignment with higher timeframe direction, reducing whipsaw.
# Volume confirmation filters low-momentum breakouts.
# Designed for low trade frequency (~15-25/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
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
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations
    range_1d = high_1d - low_1d
    close_prev = close_1d
    
    # Resistance levels
    h4 = close_prev + 1.5 * range_1d * 1.1 / 2
    h3 = close_prev + 1.25 * range_1d * 1.1 / 2
    h2 = close_prev + 1.0 * range_1d * 1.1 / 2
    h1 = close_prev + 0.75 * range_1d * 1.1 / 2
    # Support levels
    l1 = close_prev - 0.75 * range_1d * 1.1 / 2
    l2 = close_prev - 1.0 * range_1d * 1.1 / 2
    l3 = close_prev - 1.25 * range_1d * 1.1 / 2
    l4 = close_prev - 1.5 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (using previous day's values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR weekly trend turns against us
            if (close[i] < l3_aligned[i]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR weekly trend turns against us
            if (close[i] > h3_aligned[i]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes above L4 with volume confirmation AND weekly uptrend
            if (close[i] > l4_aligned[i]) and (volume[i] > vol_ma[i]) and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below H4 with volume confirmation AND weekly downtrend
            elif (close[i] < h4_aligned[i]) and (volume[i] > vol_ma[i]) and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals