#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots (weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous weekly data to avoid look-ahead (use completed weekly bar)
    high_1w_prev = df_1w['high'].shift(1).values
    low_1w_prev = df_1w['low'].shift(1).values
    close_1w_prev = df_1w['close'].shift(1).values
    
    # Calculate weekly Camarilla levels (H4/L4 breakout)
    pivot_prev = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    range_1w_prev = high_1w_prev - low_1w_prev
    h4_prev = pivot_prev + (range_1w_prev * 1.1 / 2)
    l4_prev = pivot_prev - (range_1w_prev * 1.1 / 2)
    
    # Align to daily (only available after weekly bar closes)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4_prev)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4_prev)
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50-day SMA
    close_series = pd.Series(close)
    sma_50 = close_series.rolling(window=50, min_periods=50).mean().values
    trend_up = close > sma_50
    trend_down = close < sma_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(300, n):
        # Skip if any values not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above weekly H4 with volume and uptrend
        long_signal = close[i] > h4_aligned[i] and volume_ok[i] and trend_up[i]
        # Short: break below weekly L4 with volume and downtrend
        short_signal = close[i] < l4_aligned[i] and volume_ok[i] and trend_down[i]
        
        # Exit on opposite breakout (mean reversion to pivot)
        pivot_prev_val = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_prev_val)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals