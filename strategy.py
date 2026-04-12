#!/usr/bin/env python3
"""
1h_4h_1d_camarilla_volume_filtered_v1
Hypothesis: Use daily Camarilla H3/L3 for trend direction (long above H3, short below L3).
Enter on 1h pullback to EMA20 with volume confirmation. Exit on opposite touch.
Targets 20-40 trades/year by requiring both HTF trend alignment and LTF pullback.
Works in bull/bear: trend filter avoids counter-trend trades, volume confirms breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels using PREVIOUS day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    
    # Align to 1h timeframe
    h3_1h = align_htf_to_ltf(prices, df_1d, h3)
    l3_1h = align_htf_to_ltf(prices, df_1d, l3)
    
    # 1h EMA20 for pullback entries
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_1h[i]) or np.isnan(l3_1h[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price must be above H3 for long, below L3 for short
        trend_long = close[i] > h3_1h[i]
        trend_short = close[i] < l3_1h[i]
        
        # Pullback to EMA20 with volume confirmation
        pullback_long = close[i] <= ema20[i] * 1.005  # within 0.5% of EMA20
        pullback_short = close[i] >= ema20[i] * 0.995  # within 0.5% of EMA20
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Entry conditions
        long_entry = trend_long and pullback_long and volume_filter
        short_entry = trend_short and pullback_short and volume_filter
        
        # Exit: touch opposite Camarilla level
        long_exit = close[i] < l3_1h[i]
        short_exit = close[i] > h3_1h[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_volume_filtered_v1"
timeframe = "1h"
leverage = 1.0