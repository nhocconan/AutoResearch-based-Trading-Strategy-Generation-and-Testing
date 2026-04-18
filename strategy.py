#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_1dTrend_Volume
Hypothesis: Combines daily trend (EMA50 > EMA200) with 6h Camarilla pivot breakouts (R1/S1) and volume confirmation.
Trades only in direction of daily trend to avoid counter-trend whipsaws. Designed for 15-30 trades/year.
Works in bull markets (riding uptrends) and bear markets (riding downtrends) by filtering with daily trend.
"""

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
    
    # Daily trend filter: EMA50 > EMA200 for long, EMA50 < EMA200 for short
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_uptrend = ema50 > ema200
    daily_downtrend = ema50 < ema200
    
    # Daily Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with daily uptrend and volume spike
            if close[i] > R1_6h[i] and daily_uptrend[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with daily downtrend and volume spike
            elif close[i] < S1_6h[i] and daily_downtrend[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or daily trend turns down
            if close[i] < S1_6h[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or daily trend turns up
            if close[i] > R1_6h[i] or not daily_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0