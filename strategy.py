#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels identify key intraday support/resistance. Breakouts above/below these levels with volume confirmation and higher timeframe trend alignment capture strong moves. Works in both bull (long breakouts) and bear (short breakdowns) markets. Low trade frequency (~15-30/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Camarilla pivot levels (based on previous 1d bar)
    # Calculate pivot points using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_h = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + range_h * 1.1 / 2
    r3 = pivot + range_h * 1.1 / 4
    r2 = pivot + range_h * 1.1 / 6
    r1 = pivot + range_h * 1.1 / 12
    s1 = pivot - range_h * 1.1 / 12
    s2 = pivot - range_h * 1.1 / 6
    s3 = pivot - range_h * 1.1 / 4
    s4 = pivot - range_h * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals (using Camarilla levels)
        breakout_r1 = close[i] > r1_aligned[i-1]  # Break above R1
        breakdown_s1 = close[i] < s1_aligned[i-1]  # Break below S1
        
        # 1d EMA trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Break above R1 AND bullish trend AND volume confirmation
        if breakout_r1 and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Break below S1 AND bearish trend AND volume confirmation
        elif breakdown_s1 and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (break below S1 for long, break above R1 for short)
        elif position == 1 and close[i] < s1_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r1_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals