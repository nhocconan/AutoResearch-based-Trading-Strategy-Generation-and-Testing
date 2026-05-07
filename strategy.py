#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Uses 12-hour EMA50 trend filter to improve performance in both bull and bear markets.
# Requires price to close beyond R1/S1 levels with volume spike (2.5x 30-period average).
# Includes minimum holding period of 3 bars to reduce whipsaw.
# Target: 20-30 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for calculations (HTF as specified)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels from 12h data
    camarilla_range = high_12h - low_12h
    r1 = close_12h + 1.1 * camarilla_range / 12
    s1 = close_12h - 1.1 * camarilla_range / 12
    
    # Get 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_12h, r1)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1)
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter on 4h (30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.5 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_50_12h_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > R1, above 12h EMA50 trend, volume spike
            if close[i] > r1_4h[i] and close[i] > ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < S1, below 12h EMA50 trend, volume spike
            elif close[i] < s1_4h[i] and close[i] < ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit conditions: require minimum 3 bars held
            if bars_since_entry >= 3:
                if close[i] < r1_4h[i] or close[i] < ema_50_12h_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                # Hold position for minimum period
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: require minimum 3 bars held
            if bars_since_entry >= 3:
                if close[i] > s1_4h[i] or close[i] > ema_50_12h_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                # Hold position for minimum period
                signals[i] = -0.25
    
    return signals