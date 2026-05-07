# 6h Weekly Pivot Point Trend Continuation with Volume Confirmation
# Weekly pivot points act as strong support/resistance levels. 
# Price breaking above/below weekly pivot with trend and volume confirmation
# indicates institutional participation and trend continuation.
# Works in bull markets (break above pivot in uptrend) and bear markets 
# (break below pivot in downtrend) by capturing institutional flow.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

#!/usr/bin/env python3
name = "6h_WeeklyPivot_TrendContinuation_Volume"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Get daily data for trend filter (EMA34)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Daily EMA34 trend filter
    ema_34_daily = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_34_daily)
    
    # Volume filter: current volume > 1.8x 20-period average (for 6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_daily_aligned[i]) or 
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
        trend_up = close > ema_34_daily_aligned[i]
        trend_down = close < ema_34_daily_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above weekly R1 with volume in uptrend
            if (close[i] > r1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below weekly S1 with volume in downtrend
            elif (close[i] < s1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below weekly pivot or trend changes
            if close[i] < pivot_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above weekly pivot or trend changes
            if close[i] > pivot_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points act as significant support/resistance levels where
# institutional order flow accumulates. Price breaking above/below weekly R1/S1
# levels with daily trend alignment and volume confirmation indicates strong
# institutional participation and trend continuation. Works in bull markets (break
# above pivot in uptrend) and bear markets (break below pivot in downtrend).
# Uses 6h timeframe to balance trade frequency and capture meaningful trends.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.