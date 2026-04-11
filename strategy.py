#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla Pivot reversal with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance on 12h timeframe. 
# Price rejecting these levels with volume confirms reversal. 1d EMA filter ensures trades align with daily trend.
# Works in bull/bear: buys dips in uptrend, sells rallies in downtrend. Low trade frequency (~15-30/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d candle
    # Typical price = (H + L + C) / 3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # Pivot point = typical price
    pivot_1d = typical_price_1d.values
    # R1 = C + (H - L) * 1.1/12
    r1_1d = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    # S1 = C - (H - L) * 1.1/12
    s1_1d = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    r1_1d_vals = r1_1d.values
    s1_1d_vals = s1_1d.values
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d_vals)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla pivot rejection + volume + trend alignment
        # Long: price rejects S1 support in uptrend with volume
        if close[i] <= s1_aligned[i] and close[i] > s1_aligned[i] * 0.999 and uptrend and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price rejects R1 resistance in downtrend with volume
        elif close[i] >= r1_aligned[i] and close[i] < r1_aligned[i] * 1.001 and downtrend and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price moves back to pivot level
        elif position == 1 and close[i] >= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals