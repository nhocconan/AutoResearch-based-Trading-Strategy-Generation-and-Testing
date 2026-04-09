#!/usr/bin/env python3
# 6h_donchian_1d_pivot_breakout_v1
# Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot filter for direction.
# Long when price breaks above 6h Donchian high AND 1d close > R3 pivot (bullish bias).
# Short when price breaks below 6h Donchian low AND 1d close < S3 pivot (bearish bias).
# Volume confirmation: current 6h volume > 1.5x 20-period average.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag.
# Works in bull/bear by using 1d pivot direction as regime filter.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1d_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    r4_1d = close_1d + range_1d * 1.1
    s4_1d = close_1d - range_1d * 1.1
    
    # Align 1d pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR 1d close < S3 (bearish shift)
            if close[i] <= donchian_low[i] or close_1d[i] < s3_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR 1d close > R3 (bullish shift)
            if close[i] >= donchian_high[i] or close_1d[i] > r3_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions with volume confirmation
            if volume_confirmed[i]:
                # Long: price breaks above Donchian high AND 1d close > R3 (bullish bias)
                if close[i] > donchian_high[i] and close_1d[i] > r3_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND 1d close < S3 (bearish bias)
                elif close[i] < donchian_low[i] and close_1d[i] < s3_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals