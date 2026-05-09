#!/usr/bin/env python3
# Strategy: 1h_Camarilla_Pivot_R1S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use Camarilla R1/S1 breakout on 1h for entry timing, with 4h EMA50 trend filter and 1d volume spike filter.
# This combines intraday precision with higher timeframe trend and volume confirmation to reduce false signals.
# Target: 15-30 trades/year per symbol by requiring confluence of breakout, trend, and volume.
# Works in bull/bear: Trend filter adapts direction, volume spike confirms institutional interest.

name = "1h_Camarilla_Pivot_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate average volume on 1d (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla levels on 1h using previous day's OHLC
    # We need daily OHLC from 1d data aligned to 1h
    # Extract daily open, high, low, close from 1d data
    # Align each to 1h
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 1h bar using prior day's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rng = high_1d_aligned - low_1d_aligned
    r1 = close_1d_aligned + 1.1 * rng / 12
    s1 = close_1d_aligned - 1.1 * rng / 12
    
    # Volume confirmation: current 1h volume > 1.5x 20-day average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_avg_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above 4h EMA50 + volume confirmation
            if close[i] > r1[i] and close[i] > ema_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S1 + below 4h EMA50 + volume confirmation
            elif close[i] < s1[i] and close[i] < ema_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or below 4h EMA50
            if close[i] < s1[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R1 or above 4h EMA50
            if close[i] > r1[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals