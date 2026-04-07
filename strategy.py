#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from 1-day timeframe with volume confirmation. 
Enter long at L4 level (bullish bias) with volume > 1.3x average, short at H4 level (bearish bias) with volume > 1.3x average.
Exit when price reaches opposite H6/L6 levels or reverses at H3/L3. Designed for low frequency (12-37 trades/year) 
to avoid fee drag while capturing mean reversion in ranging markets and breakouts in trending markets.
Works in bull (buy L4 bounces) and bear (sell H4 rejections) by using volume confirmation and pivot levels as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
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
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H6 = close + 2.0 * (high - low)
    # L6 = close - 2.0 * (high - low)
    range_hl = d_high - d_low
    camarilla_h4 = d_close + 1.5 * range_hl
    camarilla_l4 = d_close - 1.5 * range_hl
    camarilla_h3 = d_close + 1.125 * range_hl
    camarilla_l3 = d_close - 1.125 * range_hl
    camarilla_h6 = d_close + 2.0 * range_hl
    camarilla_l6 = d_close - 2.0 * range_hl
    
    # Align all levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to have previous day's data
        # Skip if daily data not available (first bar)
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price reaches H6 (strong resistance) or reverses below L3
            if close[i] >= h6_aligned[i] or close[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches L6 (strong support) or reverses above H3
            if close[i] <= l6_aligned[i] or close[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at L4 level with volume confirmation
            long_entry = (abs(close[i] - l4_aligned[i]) < 0.001 * close[i]) and vol_confirm
            # Short entry: price at H4 level with volume confirmation
            short_entry = (abs(close[i] - h4_aligned[i]) < 0.001 * close[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals