#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot bounce with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (based on 1d OHLC) act as strong support/resistance. 
# Price bouncing off L3/H3 levels with volume confirmation and aligned 1d trend offers high-probability entries.
# Works in both bull and bear markets by trading mean reversion within the day's range.
# Low frequency (~20-30/year) minimizes fee drag.

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
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from 1d data
    # Camarilla: based on previous day's OHLC
    # Resistance levels: H4, H3, H2, H1
    # Support levels: L1, L2, L3, L4
    # We focus on H3 (strong resistance) and L3 (strong support)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate pivots using vectorized operations
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.6 * (high - low)
    # H1 = close + 0.4 * (high - low)
    # L1 = close - 0.4 * (high - low)
    # L2 = close - 0.6 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # We need previous day's values, so shift by 1
    if len(high_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d_vals, 1)
    # First day has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    H3 = prev_close + 1.1 * (prev_high - prev_low)
    L3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla bounce + volume + trend alignment
        # Long: price near L3 support in uptrend with volume
        if (close[i] <= L3_aligned[i] * 1.005 and  # Within 0.5% of L3
            close[i] >= L3_aligned[i] * 0.995 and
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: price near H3 resistance in downtrend with volume
        elif (close[i] >= H3_aligned[i] * 0.995 and  # Within 0.5% of H3
              close[i] <= H3_aligned[i] * 1.005 and
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price moves to opposite level or trend change
        elif position == 1 and (close[i] >= H3_aligned[i] * 0.995 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= L3_aligned[i] * 1.005 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals