#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_volume_v1
Hypothesis: On daily timeframe, price touching Camarilla pivot levels (L3/L4 for long, H3/H4 for short) with volume expansion and weekly trend alignment captures reversals in both bull and bear markets. Weekly trend filter ensures we trade with the higher timeframe momentum, reducing false signals in ranging conditions.
- Long: Price <= L3 (Camarilla) + volume > 1.5x 20-day average + weekly uptrend
- Short: Price >= H3 (Camarilla) + volume > 1.5x 20-day average + weekly downtrend
- Exit: Opposite Camarilla level (H3 for long, L3 for short) or weekly trend reversal
- Position sizing: 0.25 long, -0.25 short
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    daily_range = high_1d - low_1d
    H3 = close_1d + 1.0 * daily_range
    L3 = close_1d - 1.0 * daily_range
    H4 = close_1d + 1.5 * daily_range
    L4 = close_1d - 1.5 * daily_range
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align daily levels and weekly trend to daily bars
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: daily volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches H3 OR weekly trend turns down
            if (close[i] >= H3_aligned[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price touches L3 OR weekly trend turns up
            if (close[i] <= L3_aligned[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price <= L3 (or L4 for stronger signal) + volume + weekly uptrend
            if (close[i] <= L3_aligned[i]) and volume_filter[i] and trend_1w_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price >= H3 (or H4 for stronger signal) + volume + weekly downtrend
            elif (close[i] >= H3_aligned[i]) and volume_filter[i] and trend_1w_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals