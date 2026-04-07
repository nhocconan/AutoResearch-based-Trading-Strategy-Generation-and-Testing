#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels for mean reversion in choppy markets.
Long when price touches S3 level with volume > 2x average and closes above it.
Short when price touches R3 level with volume > 2x average and closes below it.
Exit when price reaches the opposite pivot level or closes at daily VWAP.
Works in both bull/bear markets as Camarilla adapts to volatility and volume confirmation filters false signals.
Designed for ~20-30 trades/year to minimize fee drag while capturing mean reversion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timezone = "12h"
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
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily VWAP for exit
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    
    # Calculate Camarilla pivot levels for each day
    # Camarilla: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    hl_range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.0 * hl_range_1d  # H3
    r2_1d = close_1d + 0.5 * hl_range_1d  # H2
    r1_1d = close_1d + 0.25 * hl_range_1d  # H1
    s1_1d = close_1d - 0.25 * hl_range_1d  # L1
    s2_1d = close_1d - 0.5 * hl_range_1d  # L2
    s3_1d = close_1d - 1.0 * hl_range_1d  # L3
    
    # Align all levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    vwap_12h = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Volume filter: 24-period average (2 days of 12h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(vwap_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: at least 2x average
        vol_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 level or closes at VWAP
            if close[i] <= s1_12h[i] or close[i] >= vwap_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 level or closes at VWAP
            if close[i] >= r1_12h[i] or close[i] <= vwap_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price touches S3 level and closes above it
                if close[i] <= s3_12h[i] and close[i] > s3_12h[i] * 0.999:  # touched S3
                    # Additional confirmation: price must close above the touch point
                    if close[i] > low[i]:  # closed above low of the bar
                        position = 1
                        signals[i] = 0.25
                # Short: price touches R3 level and closes below it
                elif close[i] >= r3_12h[i] and close[i] < r3_12h[i] * 1.001:  # touched R3
                    # Additional confirmation: price must close below the touch point
                    if close[i] < high[i]:  # closed below high of the bar
                        position = -1
                        signals[i] = -0.25
    
    return signals