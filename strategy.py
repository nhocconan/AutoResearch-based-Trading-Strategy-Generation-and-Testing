#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly trend filter and weekly pivot breakout
# - Weekly trend: price above/below weekly EMA(20) determines long/short bias
# - Entry: price breaks above/below weekly pivot R1/S1 levels with volume confirmation
# - Volume filter: current 6h volume > 1.5x 20-period average
# - Exit: opposite pivot level touch or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following weekly trend
# - Target: 15-25 trades/year to avoid excessive fee drift

name = "6h_WeeklyTrend_PivotBreakout_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot levels
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA(20) for trend direction
    ema_20_w = pd.Series(df_w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_w_aligned = align_htf_to_ltf(prices, df_w, ema_20_w)
    
    # Weekly pivot points (using typical price)
    typical_price_w = (df_w['high'].values + df_w['low'].values + df_w['close'].values) / 3
    pivot_w = (df_w['high'].values + df_w['low'].values + df_w['close'].values) / 3
    r1_w = 2 * pivot_w - df_w['low'].values
    s1_w = 2 * pivot_w - df_w['high'].values
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > weekly EMA20) + break above R1 + volume
            if close[i] > ema_20_w_aligned[i] and high[i] > r1_w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < weekly EMA20) + break below S1 + volume
            elif close[i] < ema_20_w_aligned[i] and low[i] < s1_w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on touch of S1 or trend reversal
            if low[i] <= s1_w_aligned[i] or close[i] < ema_20_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on touch of R1 or trend reversal
            if high[i] >= r1_w_aligned[i] or close[i] > ema_20_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals