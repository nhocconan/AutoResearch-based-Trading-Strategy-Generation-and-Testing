#!/usr/bin/env python3
# 6h_WeeklyPivot_R4S4_Breakout_1dTrend_Volume
# Hypothesis: Weekly pivot levels (R4/S4) on 1-day chart identify extreme support/resistance. 
# Price breaking above R4 in a daily uptrend or below S4 in a daily downtrend signals strong momentum.
# Volume confirmation filters false breakouts. Weekly pivots adapt to volatility, working in both bull and bear markets.

name = "6h_WeeklyPivot_R4S4_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get daily data for weekly pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points using daily high/low/close (weekly = 5 trading days)
    # We use the last 5 daily candles to compute weekly pivot
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r4 = weekly_pivot + 3 * (weekly_high - weekly_low)  # R4 = P + 3*(H-L)
    weekly_s4 = weekly_pivot - 3 * (weekly_high - weekly_low)  # S4 = P - 3*(H-L)
    
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34), weekly pivot (5), and volume MA (20)
    start_idx = max(34, 5, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(weekly_r4_aligned[i]) or 
            np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (2.0x MA to reduce false signals)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R4 + volume
            if uptrend and close[i] > weekly_r4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S4 + volume
            elif downtrend and close[i] < weekly_s4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below weekly R4
            if not uptrend or close[i] < weekly_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above weekly S4
            if not downtrend or close[i] > weekly_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals