#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot with 1d Trend and Volume Filter v1
# Hypothesis: Camarilla pivot levels (L3, L4, H3, H4) from daily timeframe act as strong support/resistance.
# In an uptrend (price > 1-day EMA), long at L3/H3 bounce; in downtrend, short at H3/L3 rejection.
# Volume confirmation ensures institutional interest. Works in both bull (trend-following) and bear (mean reversion at extremes).
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_camarilla_pivot_1d_trend_volume_v1"
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
    
    # Get 1d data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Using typical Camarilla formulas based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # Previous day close
    
    # Avoid look-ahead: shift by 1 to use only completed daily bars
    range_1d = high_1d - low_1d
    # Camarilla levels
    H3 = close_1d_prev + range_1d * 1.1 / 6
    L3 = close_1d_prev - range_1d * 1.1 / 6
    H4 = close_1d_prev + range_1d * 1.1 / 2
    L4 = close_1d_prev - range_1d * 1.1 / 2
    
    # Align to 4h timeframe (already shifted by 1 in calculation)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 or trend turns down
            if close[i] < L3_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above H3 or trend turns up
            if close[i] > H3_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume must be above average
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
                
            # Long: price > 1d EMA (uptrend) and price touches/bounces from L3
            if close[i] > ema_1d_aligned[i]:
                # Allow small buffer for entry: within 0.1% of L3
                if abs(close[i] - L3_aligned[i]) / L3_aligned[i] < 0.001:
                    position = 1
                    signals[i] = 0.25
            # Short: price < 1d EMA (downtrend) and price touches/rejects from H3
            elif close[i] < ema_1d_aligned[i]:
                # Allow small buffer for entry: within 0.1% of H3
                if abs(close[i] - H3_aligned[i]) / H3_aligned[i] < 0.001:
                    position = -1
                    signals[i] = -0.25
    
    return signals