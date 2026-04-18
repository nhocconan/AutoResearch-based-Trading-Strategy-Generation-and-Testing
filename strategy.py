#!/usr/bin/env python3
"""
Hypothesis: 6s-based strategy using weekly pivot levels (R1/S1) from 1w combined with 1d EMA(50) trend filter and volume confirmation. 
Weekly pivots provide strong support/resistance in trending and ranging markets while the 1d EMA filters for trend direction.
Volume confirmation ensures breakouts have conviction. Designed for 15-30 trades/year to minimize fee drag.
Works in bull markets (buy R1 breaks in uptrend) and bear markets (sell S1 breaks in downtrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels (R1, S1) from previous week
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    weekly_r1 = np.full(len(close_1w), np.nan)
    weekly_s1 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        if not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1]) or np.isnan(close_1w[i-1])):
            weekly_r1[i] = close_1w[i-1] + 1.1 * (high_1w[i-1] - low_1w[i-1]) / 12
            weekly_s1[i] = close_1w[i-1] - 1.1 * (high_1w[i-1] - low_1w[i-1]) / 12
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2/51) + (ema_50_1d[i-1] * 49/51)
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume moving average (24-period, equivalent to 6 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # need EMA, weekly pivot, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8 * 24-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_6h[i]
        trend_down = close[i] < ema_50_1d_6h[i]
        
        if position == 0:
            # Long entry: close above weekly R1 with volume and uptrend
            if (close[i] > weekly_r1_6h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below weekly S1 with volume and downtrend
            elif (close[i] < weekly_s1_6h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below weekly S1 or reverse signal
            if close[i] < weekly_s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly R1 or reverse signal
            if close[i] > weekly_r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0