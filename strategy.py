#!/usr/bin/env python3
name = "6h_WeeklyPivot_Momentum_Breakout_v2"
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
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    prev_week_high = df_weekly['high'].shift(1).values
    prev_week_low = df_weekly['low'].shift(1).values
    prev_week_close = df_weekly['close'].shift(1).values
    
    # Weekly pivot points (standard formula)
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Get daily trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily EMA20 trend filter
    ema_20_daily = pd.Series(df_daily['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Volume filter: current volume > 1.5x 6-period average (24h for 6h)
    vol_ma_6 = np.full(n, np.nan)
    for i in range(6, n):
        vol_ma_6[i] = np.mean(volume[i-6:i])
    vol_filter = volume > (1.5 * vol_ma_6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(50, 6, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_20_daily_aligned[i]) or 
            np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close > ema_20_daily_aligned[i]
        trend_down = close < ema_20_daily_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R1 with volume in uptrend
            if (close[i] > r1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S1 with volume in downtrend
            elif (close[i] < s1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below S1 or trend changes
            if close[i] < s1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above R1 or trend changes
            if close[i] > r1_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot breakout with daily EMA20 trend filter and volume confirmation.
# Long when price breaks above weekly R1 in daily uptrend with volume confirmation.
# Short when price breaks below weekly S1 in daily downtrend with volume confirmation.
# Uses 6h timeframe for balance of signal quality and trade frequency.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Weekly pivot provides strong institutional levels, daily trend filters false breakouts.