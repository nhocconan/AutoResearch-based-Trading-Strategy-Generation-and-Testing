#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla pivot levels with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide strong support/resistance. Price breaking above/below these levels with volume confirmation and aligned with daily trend (price above/below EMA50) indicates momentum continuation. Works in both bull and bear markets by only trading in direction of higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas:
    # H5 = close + 1.1*(high-low)*1.5
    # H4 = close + 1.1*(high-low)*1.25
    # H3 = close + 1.1*(high-low)*1.0
    # L3 = close - 1.1*(high-low)*1.0
    # L4 = close - 1.1*(high-low)*1.25
    # L5 = close - 1.1*(high-low)*1.5
    # We'll use H3, L3 for breakout and H4, L4 for stronger breakout
    
    daily_range = high_1d - low_1d
    camarilla_H3 = close_1d + 1.1 * daily_range * 1.0
    camarilla_L3 = close_1d - 1.1 * daily_range * 1.0
    camarilla_H4 = close_1d + 1.1 * daily_range * 1.25
    camarilla_L4 = close_1d - 1.1 * daily_range * 1.25
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        # Long: break above H3 or H4 with volume and uptrend
        if ((close[i] > camarilla_H3_aligned[i] or close[i] > camarilla_H4_aligned[i]) and
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: break below L3 or L4 with volume and downtrend
        elif ((close[i] < camarilla_L3_aligned[i] or close[i] < camarilla_L4_aligned[i]) and
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to midpoint or trend changes
        elif position == 1 and (close[i] < camarilla_H3_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_L3_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals