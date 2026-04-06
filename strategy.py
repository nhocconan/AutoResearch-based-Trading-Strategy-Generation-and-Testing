#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume spike and 1w trend filter
# Enter long at L3 support when: price crosses above L3, volume > 2x average, price > 1w EMA(50)
# Enter short at H3 resistance when: price crosses below H3, volume > 2x average, price < 1w EMA(50)
# Exit when price reaches opposite H3/L3 level or opposite Camarilla level (H4/L4)
# Targets 80-150 trades over 4 years by combining intraday reversals with weekly trend filter

name = "4h_camarilla_pivot_1w_trend_vol_v1"
timeframe = "4h"
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
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_calc = lambda c, h, l: (
        c + 1.5 * (h - l),  # H4
        c + 1.1 * (h - l),  # H3
        c - 1.1 * (h - l),  # L3
        c - 1.5 * (h - l)   # L4
    )
    
    # Calculate Camarilla levels for each 1d bar
    h4_1d, h3_1d, l3_1d, l4_1d = camarilla_calc(close_1d, high_1d, low_1d)
    
    # Align to 4h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price reaches H3 (profit target) or breaks below L4 (stop)
            if close[i] >= h3_1d_aligned[i] or close[i] <= l4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches L3 (profit target) or breaks above H4 (stop)
            if close[i] <= l3_1d_aligned[i] or close[i] >= h4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price crosses Camarilla levels with volume and trend filter
            if volume[i] > volume_threshold[i]:
                # Long: price crosses above L3 from below AND above weekly EMA
                if close[i] > l3_1d_aligned[i] and close[i-1] <= l3_1d_aligned[i-1] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price crosses below H3 from above AND below weekly EMA
                elif close[i] < h3_1d_aligned[i] and close[i-1] >= h3_1d_aligned[i-1] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals