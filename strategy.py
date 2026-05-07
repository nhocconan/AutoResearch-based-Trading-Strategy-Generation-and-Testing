#!/usr/bin/env python3
"""
6H_WeeklyPivot_Trend_Range_Switch_v1
Hypothesis: Use weekly pivot levels as structural anchors and switch strategy based on weekly range.
- In trending weeks (weekly range > 200% of ATR(20)): breakout of weekly R1/S1 with trend filter.
- In ranging weeks (weekly range <= 200% of ATR(20)): mean reversion at weekly S2/R2 with reversal signals.
Uses 1d EMA34 for trend filter in trending mode and RSI(14) for mean reversion in ranging mode.
Designed to work in both bull and bear markets by adapting to weekly volatility regime.
"""
name = "6H_WeeklyPivot_Trend_Range_Switch_v1"
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
    
    # Get weekly data for pivot levels and regime detection
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly pivot levels (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    r1 = weekly_pivot + (weekly_range * 1.1 / 12)
    r2 = weekly_pivot + (weekly_range * 1.1 / 6)
    s1 = weekly_pivot - (weekly_range * 1.1 / 12)
    s2 = weekly_pivot - (weekly_range * 1.1 / 6)
    
    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Weekly range for regime detection
    weekly_range_pct = weekly_range / weekly_pivot
    weekly_range_aligned = align_htf_to_ltf(prices, df_weekly, weekly_range_pct)
    
    # Get daily EMA34 for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    
    # RSI(14) for mean reversion signals
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 10)  # Ensure EMA and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(weekly_range_aligned[i]) or np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_range_val = weekly_range_aligned[i]
        is_trending = weekly_range_val > 2.0  # Weekly range > 200% of pivot
        
        if position == 0:
            if is_trending:
                # Trending week: breakout of R1/S1 with daily EMA34 filter
                if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                    close[i] > ema_34_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                      close[i] < ema_34_aligned[i]):
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging week: mean reversion at S2/R2 with RSI extremes
                if (close[i] <= s2_aligned[i] and rsi_values[i] < 30):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] >= r2_aligned[i] and rsi_values[i] > 70):
                    signals[i] = -0.25
                    position = -1
        elif position != 0:
            # Exit conditions
            if position == 1:
                if is_trending:
                    # Exit on return to weekly pivot or below EMA34
                    if close[i] <= weekly_pivot[min(i // (7*24//6), len(weekly_pivot)-1)] if i >= 7*24//6 else False or close[i] < ema_34_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                else:
                    # Exit on RSI reversal or at R1
                    if rsi_values[i] > 50 or close[i] >= r1_aligned[i]:
                        signals[i] = 0.0
                        position = 0
            elif position == -1:
                if is_trending:
                    # Exit on return to weekly pivot or above EMA34
                    if close[i] >= weekly_pivot[min(i // (7*24//6), len(weekly_pivot)-1)] if i >= 7*24//6 else False or close[i] > ema_34_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                else:
                    # Exit on RSI reversal or at S1
                    if rsi_values[i] < 50 or close[i] <= s1_aligned[i]:
                        signals[i] = 0.0
                        position = 0
    
    return signals