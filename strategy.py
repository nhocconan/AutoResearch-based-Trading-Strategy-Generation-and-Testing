#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation
# 4h/1d pivots provide structure aligned with 1h timeframe
# Volume confirmation (current 1h volume > 2.0x 20-period average) filters false breakouts
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Works in bull/bear: price reacts to 4h/1d structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.20 to minimize fee churn

name = "1h_4h_1d_camarilla_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) - prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 25 or len(df_1d) < 25:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla pivot levels
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    camarilla_r4_4h = close_4h + range_4h * 1.1 / 2.0  # R4
    camarilla_s4_4h = close_4h - range_4h * 1.1 / 2.0  # S4
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2.0  # R4
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2.0  # S4
    
    # Align Camarilla levels to 1h timeframe
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 2.0x average 1h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on 4h S4 retracement (mean reversion from strong level)
            if close[i] < s4_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit on 4h R4 retracement (mean reversion from strong level)
            if close[i] > r4_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout trading with volume confirmation AND 1d level alignment
            # Require both 4h breakout AND price beyond 1d level for stronger signal
            if volume_confirmed:
                # Long on 4h R4 breakout AND price above 1d R4
                if close[i] > r4_4h_aligned[i] and close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short on 4h S4 breakout AND price below 1d S4
                elif close[i] < s4_4h_aligned[i] and close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals