#!/usr/bin/env python3
name = "6h_1w_Pivot_Trend_Filter_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (Monday close as pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points using Monday's close
    # Pivot = Monday close, R1 = Pivot + (Monday High - Monday Low), S1 = Pivot - (Monday High - Monday Low)
    # We'll use the Monday of each week as the reference point
    # For simplicity, we'll use weekly high/low/close to calculate pivots
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point (standard formula)
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly R1 and S1
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily data for trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA 34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h to reduce trades
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R1 in uptrend with volume
            if (close[i] > r1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S1 in downtrend with volume
            elif (close[i] < s1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters between S1 and R1 or trend change
            if (close[i] > s1_aligned[i] and close[i] < r1_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters between S1 and R1 or trend change
            if (close[i] > s1_aligned[i] and close[i] < r1_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points (R1/S1) provide significant support/resistance levels.
# Breaking above R1 in a daily uptrend with volume confirmation captures institutional
# buying, while breaking below S1 in a daily downtrend captures selling.
# The 6h timeframe reduces noise compared to lower timeframes, and weekly pivots
# are less prone to false breaks than daily levels. This should work in both bull
# and bear markets as it follows the trend with institutional confirmation.
# Target: 20-40 trades/year to minimize fee drag.