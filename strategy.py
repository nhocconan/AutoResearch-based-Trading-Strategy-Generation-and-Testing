#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s timeframe with weekly trend filter and daily pivot breakout
# - Weekly trend: price > weekly EMA(21) for long bias, price < weekly EMA(21) for short bias
# - Daily Pivot: calculate from daily OHLC; R1/S1 as breakout levels, R2/S2 as stronger breakout
# - Entry: price breaks above R1 in uptrend (long) or below S1 in downtrend (short)
# - Exit: price returns to daily pivot level or trend reversal
# - Volume filter: current 6s volume > 1.5x 20-period average of 6s volume
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "6h_WeeklyTrend_DailyPivotBreakout_Volume_v1"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend direction
    ema_21_weekly = pd.Series(df_weekly['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_21_weekly)
    
    # Get daily data for pivot points
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    
    # Align pivot levels to 6s timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    
    # Volume filter: current 6s volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_weekly_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long entry: uptrend (price > weekly EMA21) + price breaks above R1 + volume
            if (close[i] > ema_21_weekly_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend (price < weekly EMA21) + price breaks below S1 + volume
            elif (close[i] < ema_21_weekly_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot or trend reverses
            if (close[i] <= pivot_aligned[i] or 
                close[i] < ema_21_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot or trend reverses
            if (close[i] >= pivot_aligned[i] or 
                close[i] > ema_21_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals