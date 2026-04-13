#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels (daily)
    # Pivot = (H + L + C) / 3
    # H4 = Pivot + 1.5 * (H - L)
    # L4 = Pivot - 1.5 * (H - L)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h4_1d = pivot_1d + 1.5 * range_1d
    l4_1d = pivot_1d - 1.5 * range_1d
    
    # Align daily data to 4h
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Align weekly close to 4h for trend filter
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        volume_condition = volume_1d_aligned[i] > (volume_ma_20_1d_aligned[i] * 1.5)
        
        # Trend filter: only long when price > weekly close, short when price < weekly close
        long_trend = close[i] > close_1w_aligned[i]
        short_trend = close[i] < close_1w_aligned[i]
        
        # Entry conditions: price at Camarilla H4 or L4 with volume and trend
        at_h4 = abs(close[i] - h4_1d_aligned[i]) < (0.001 * close[i])  # within 0.1% of H4
        at_l4 = abs(close[i] - l4_1d_aligned[i]) < (0.001 * close[i])  # within 0.1% of L4
        
        if position == 0:
            if at_h4 and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            elif at_l4 and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price moves back above L4 (mean reversion)
            if close[i] > l4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price moves back below H4 (mean reversion)
            if close[i] < h4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Reversion"
timeframe = "4h"
leverage = 1.0