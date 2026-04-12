#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Use previous 12h bar's data to avoid look-ahead
    high_12h_prev = df_12h['high'].shift(1).values
    low_12h_prev = df_12h['low'].shift(1).values
    close_12h_prev = df_12h['close'].shift(1).values
    
    # Calculate Camarilla pivot levels from previous 12h data
    pivot_prev = (high_12h_prev + low_12h_prev + close_12h_prev) / 3.0
    range_12h_prev = high_12h_prev - low_12h_prev
    
    # Camarilla levels (H4 and L4 - breakout levels)
    h4_prev = pivot_prev + (range_12h_prev * 1.1 / 2)
    l4_prev = pivot_prev - (range_12h_prev * 1.1 / 2)
    
    # Align levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4_prev)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4_prev)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50-period SMA on 4h data (slower MA for stronger trend)
    close_series = pd.Series(close)
    sma_50 = close_series.rolling(window=50, min_periods=50).mean().values
    trend_up = close > sma_50
    trend_down = close < sma_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above H4 with volume confirmation and uptrend
        long_signal = close[i] > h4_aligned[i] and volume_ok[i] and trend_up[i]
        # Short: price breaks below L4 with volume confirmation and downtrend
        short_signal = close[i] < l4_aligned[i] and volume_ok[i] and trend_down[i]
        
        # Exit when price returns to pivot (mean reversion)
        pivot_prev_val = (high_12h_prev + low_12h_prev + close_12h_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_prev_val)
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