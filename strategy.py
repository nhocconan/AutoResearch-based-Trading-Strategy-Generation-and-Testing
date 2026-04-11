#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_reversion_v1
# Strategy: 4h Camarilla pivot mean reversion with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price tends to revert to Camarilla pivot levels (H3/L3) during ranging markets.
# In trending markets (1d EMA50), we trade pullbacks to H4/L4 in direction of trend.
# Volume > 1.3x 20-period average confirms institutional participation.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.
# Works in bull markets via long pullbacks to L4 and bear markets via short pullbacks to H4.
# Uses Camarilla levels calculated from prior 1d session.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_reversion_v1"
timeframe = "4h"
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
    
    # Calculate Camarilla levels from prior 1d session
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # Where H,L,C are from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan  # First value invalid
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each 1d bar
    H4_1d = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 2
    L4_1d = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 2
    H3_1d = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 4
    L3_1d = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 4
    
    # Align to 4h timeframe (wait for 1d bar to close)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H4_1d_aligned[i]) or np.isnan(L4_1d_aligned[i]) or 
            np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Trend filter: price above EMA = bullish, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price touches L3 or L4 AND bullish trend AND volume confirmation
        long_touch = (low[i] <= L3_1d_aligned[i] or low[i] <= L4_1d_aligned[i])
        if long_touch and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price touches H3 or H4 AND bearish trend AND volume confirmation
        elif (high[i] >= H3_1d_aligned[i] or high[i] >= H4_1d_aligned[i]) and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price reaches opposite level (H4 for long, L4 for short) or opposite touch
        elif position == 1 and (high[i] >= H4_1d_aligned[i] or low[i] <= L3_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= L4_1d_aligned[i] or high[i] >= H3_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals