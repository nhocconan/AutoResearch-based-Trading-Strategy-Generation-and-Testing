#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Use Camarilla pivot points (R1/S1) on 4h as support/resistance, with 4h EMA50 trend filter and volume confirmation. Trade only on 1h when price breaks above R1 in 4h uptrend or below S1 in 4h downtrend. This reduces false signals by requiring higher timeframe alignment. Session filter (08-20 UTC) further reduces noise. Target: 60-150 total trades over 4 years.
"""
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C = (H+L+CLOSE)/3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    hl_range = df_4h['high'] - df_4h['low']
    r1_4h = typical_price + hl_range * 1.1 / 12
    s1_4h = typical_price - hl_range * 1.1 / 12
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h.values)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h.values)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready or outside session
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume
            if close[i] > r1_4h_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume
            elif close[i] < s1_4h_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to the opposite Camarilla level (mean reversion)
            if position == 1:
                if close[i] < s1_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > r1_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals