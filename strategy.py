#!/usr/bin/env python3
"""
4h_MultiTimeframe_Pivot_Zone_MeanReversion
4h strategy using 12h/1d pivot zones with volume confirmation and mean reversion logic.
- Long: Price touches daily S1 pivot + 12h trend bullish + volume spike (2x avg)
- Short: Price touches daily R1 pivot + 12h trend bearish + volume spike (2x avg)
- Exit: Price reaches opposite pivot level or trend reversal
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in ranging markets (mean reversion at pivot levels) and trending markets (pullbacks to pivot)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivots to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from 12h EMA34
        # Use previous bar's EMA to avoid look-ahead (trend confirmation from closed 12h bar)
        if i >= 1:
            trend_bullish = ema_34_12h_aligned[i-1] > close[i-1]  # price above EMA = bullish
            trend_bearish = ema_34_12h_aligned[i-1] < close[i-1]  # price below EMA = bearish
        else:
            trend_bullish = False
            trend_bearish = False
        
        # Volume confirmation (2x average)
        vol_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # Price touching pivot zones (with small tolerance)
        touch_s1 = low[i] <= s1_aligned[i] * 1.002 and high[i] >= s1_aligned[i] * 0.998
        touch_r1 = high[i] >= r1_aligned[i] * 0.998 and low[i] <= r1_aligned[i] * 1.002
        
        if position == 0:
            # Long: touch S1 + bullish 12h trend + volume spike
            if touch_s1 and trend_bullish and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: touch R1 + bearish 12h trend + volume spike
            elif touch_r1 and trend_bearish and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: touch R1 or trend turns bearish
            if touch_r1 or not trend_bullish:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: touch S1 or trend turns bullish
            if touch_s1 or not trend_bearish:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MultiTimeframe_Pivot_Zone_MeanReversion"
timeframe = "4h"
leverage = 1.0