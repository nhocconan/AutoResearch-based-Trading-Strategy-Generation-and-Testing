#!/usr/bin/env python3
# 1d_Camarilla_Pivot_R4S4_Breakout_1wTrend_Volume
# Hypothesis: On daily timeframe, use weekly EMA50 for trend filter and daily Camarilla R4/S4 levels for breakout entries.
# Enter long when price breaks above R4 with volume confirmation and weekly uptrend.
# Enter short when price breaks below S4 with volume confirmation and weekly downtrend.
# Exit when price returns to the daily pivot point.
# Weekly trend filter avoids counter-trend trades; volume confirmation ensures momentum.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in bull markets via R4 breakouts and in bear markets via S4 breakdowns with trend alignment.

name = "1d_Camarilla_Pivot_R4S4_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Load daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Typical price (Pivot point)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla levels (using R4 and S4 for breakouts)
    r4 = daily_pivot + daily_range * 1.5
    s4 = daily_pivot - daily_range * 1.5
    
    # Align daily levels to 1d timeframe (with 1-bar delay for completed daily bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(weekly_ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        pivot_val = pivot_aligned[i]
        weekly_trend = weekly_ema50_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above R4 with volume confirmation and weekly uptrend
            if close[i] > r4_val and close[i] > weekly_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with volume confirmation and weekly downtrend
            elif close[i] < s4_val and close[i] < weekly_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion)
            if close[i] <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion)
            if close[i] >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals