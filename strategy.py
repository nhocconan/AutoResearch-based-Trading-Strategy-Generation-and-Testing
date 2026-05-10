#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Confluence_Strategy
Hypothesis: Uses weekly pivot points (from weekly high/low/close) as key support/resistance levels.
In bull markets, buy pullbacks to weekly pivot/S1 in uptrend; in bear markets, sell rallies to weekly pivot/R1 in downtrend.
Adds volume confirmation and 12h EMA filter to avoid false breaks. Designed for low frequency (15-25 trades/year)
to work in both trending and ranging markets by fading extremes and catching reversals.
"""

name = "6h_Weekly_Pivot_Confluence_Strategy"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or \
           np.isnan(weekly_r1_aligned[i]) or np.isnan(ema_12h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirm = volume[i] > 1.3 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price near weekly S1/S2 in uptrend (above 12h EMA50) with volume
            if (close[i] <= weekly_s1_aligned[i] * 1.005 or close[i] <= weekly_s2_aligned[i] * 1.005) and \
               close[i] > ema_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price near weekly R1/R2 in downtrend (below 12h EMA50) with volume
            elif (close[i] >= weekly_r1_aligned[i] * 0.995 or close[i] >= weekly_r2_aligned[i] * 0.995) and \
                 close[i] < ema_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches weekly pivot or R1, or breaks below 12h EMA50
            if close[i] >= weekly_pivot_aligned[i] * 0.995 or close[i] >= weekly_r1_aligned[i] * 0.995 or \
               close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches weekly pivot or S1, or breaks above 12h EMA50
            if close[i] <= weekly_pivot_aligned[i] * 1.005 or close[i] <= weekly_s1_aligned[i] * 1.005 or \
               close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals