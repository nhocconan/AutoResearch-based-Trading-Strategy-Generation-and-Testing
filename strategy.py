#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels (H3/L3) with volume confirmation
# Long when price touches or breaks above 1w H3 level AND volume > 1.5 * avg_volume(20) on 1d
# Short when price touches or breaks below 1w L3 level AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price returns to 1w Pivot level (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Camarilla H3/L3 represent strong resistance/support levels from weekly structure
# Volume confirmation ensures breakout validity while limiting false signals
# Pivot reversion exit provides logical profit target in ranging markets
# Works in both bull (buy H3 breakouts) and bear (sell L3 breakdowns) markets

name = "1d_1wCamarillaH3L3_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 completed weekly bars for pivot calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels: H3, L3, Pivot
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    # Pivot = (High + Low + Close) / 3
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) / 2
    camarilla_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation, in session
            if close[i] > h3_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation, in session
            elif close[i] < l3_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Pivot level (mean reversion)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Pivot level (mean reversion)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals