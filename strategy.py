#!/usr/bin/env python3
# 12h_camarilla_volume_breakout_v1
# Hypothesis: 12-hour Camarilla pivot levels act as support/resistance in ranging markets,
# with breakouts confirmed by volume surges. Works in both bull/bear via mean reversion
# at extreme levels and breakout continuation. Uses weekly trend filter to avoid
# counter-trend trades. Target: 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA(50) on weekly closes
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = Close + 1.5*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # H2 = Close + 0.5*(High-Low)
    # L2 = Close - 0.5*(High-Low)
    # H1 = Close + 0.25*(High-Low)
    # L1 = Close - 0.25*(High-Low)
    # We'll use H3/L3 for breakouts and H4/L4 as extreme levels
    
    camarilla_calc = []
    for i in range(len(close_1d)):
        hl = high_1d[i] - low_1d[i]
        if hl <= 0:
            camarilla_calc.append([close_1d[i]] * 8)  # fallback if no range
        else:
            camarilla_calc.append([
                close_1d[i] + 1.5 * hl,  # H4
                close_1d[i] + 1.125 * hl, # H3
                close_1d[i] + 0.5 * hl,   # H2
                close_1d[i] + 0.25 * hl,  # H1
                close_1d[i] - 0.25 * hl,  # L1
                close_1d[i] - 0.5 * hl,   # L2
                close_1d[i] - 1.125 * hl, # L3
                close_1d[i] - 1.5 * hl    # L4
            ])
    
    camarilla_array = np.array(camarilla_calc)
    h4_1d = camarilla_array[:, 0]
    h3_1d = camarilla_array[:, 1]
    l3_1d = camarilla_array[:, 6]
    l4_1d = camarilla_array[:, 7]
    
    # Align Camarilla levels to 12h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume confirmation: 12h volume > 1.8x 20-period average
    vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (1.8 * vol_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 or weekly trend turns down
            if close[i] < l3_1d_aligned[i] or (not weekly_uptrend and weekly_downtrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 or weekly trend turns up
            if close[i] > h3_1d_aligned[i] or (not weekly_downtrend and weekly_uptrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above H3 with volume surge in uptrend
            if (close[i] > h3_1d_aligned[i] and 
                weekly_uptrend and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume surge in downtrend
            elif (close[i] < l3_1d_aligned[i] and 
                  weekly_downtrend and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals