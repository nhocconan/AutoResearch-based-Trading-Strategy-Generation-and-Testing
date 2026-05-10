# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 6h_Range_Trader_Pivot_Bounce_1dTrend
# Hypothesis: Trade reversals from daily pivot points (PP, R1, S1) on 6h timeframe
# with 1d trend filter. In ranging markets (common in 2025), price tends to
# revert to daily pivot levels. Strong trend filter prevents counter-trend trades.
# Uses tight entry/exit rules to limit trades to 12-30 per year per symbol.
# Works in both bull/bear by aligning with 1d trend while capturing mean reversion.

name = "6h_Range_Trader_Pivot_Bounce_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    r1 = (2 * pivot - df_1d['low'].values)
    s1 = (2 * pivot - df_1d['high'].values)
    
    # Align pivot points to 6h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 24-period MA on 6h (1 day of 6h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA(34) and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (at least average volume)
        volume_confirm = volume[i] > volume_ma[i] * 0.8
        
        # Distance to pivot levels (as fraction of price)
        dist_to_pivot = abs(close[i] - pivot_aligned[i]) / close[i]
        dist_to_r1 = abs(close[i] - r1_aligned[i]) / close[i]
        dist_to_s1 = abs(close[i] - s1_aligned[i]) / close[i]
        
        # Entry conditions: bounce from S1 in uptrend or R1 in downtrend
        # Only trade when price is near support/resistance AND trend aligns
        near_s1 = dist_to_s1 < 0.003  # within 0.3% of S1
        near_r1 = dist_to_r1 < 0.003  # within 0.3% of R1
        near_pivot = dist_to_pivot < 0.002  # within 0.2% of pivot (exit zone)
        
        if position == 0:
            # Long: price near S1 in uptrend with volume
            if near_s1 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price near R1 in downtrend with volume
            elif near_r1 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or trend breaks
            if near_pivot or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or trend breaks
            if near_pivot or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals