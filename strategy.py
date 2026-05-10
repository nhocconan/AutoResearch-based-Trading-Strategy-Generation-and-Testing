#!/usr/bin/env python3
# 12h_Camarilla_Pivot_MeanReversion_With_DailyTrend
# Hypothesis: Mean reversion at Camarilla pivot levels (S3/S4 for long, R3/R4 for short) 
# using daily pivot levels on 12h timeframe. Only trade in direction of daily trend 
# (EMA34) to avoid counter-trend trades. Volume confirmation reduces false signals.
# Works in bull/bear by following daily trend while exploiting intraday mean reversion.
# Target: 20-35 trades/year per symbol.

name = "12h_Camarilla_Pivot_MeanReversion_With_DailyTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Calculate Camarilla levels from previous day
    # We need previous day's OHLC for current day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)  # Previous day close
    close_1d_prev[0] = close_1d[0]  # First day uses its own close
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.zeros(len(close_1d))
    camarilla_S4 = np.zeros(len(close_1d))
    camarilla_R3 = np.zeros(len(close_1d))
    camarilla_R4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        # Use previous day's data to calculate today's levels
        if i == 0:
            # First day: use same day's data (no previous)
            day_high = high_1d[i]
            day_low = low_1d[i]
            day_close = close_1d[i]
        else:
            day_high = high_1d[i-1]
            day_low = low_1d[i-1]
            day_close = close_1d[i-1]
        
        range_val = day_high - day_low
        camarilla_S3[i] = day_close - 1.0416 * range_val
        camarilla_S4[i] = day_close - 1.5 * range_val
        camarilla_R3[i] = day_close + 1.0416 * range_val
        camarilla_R4[i] = day_close + 1.5 * range_val
    
    # Align Camarilla levels to 12h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5  # Reduced from 2.0 to increase signal frequency slightly
        
        trend_up = trend_1d_up_aligned[i] > 0.5
        trend_down = trend_1d_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price at or below S3/S4 + daily uptrend + volume
            if (close[i] <= S3_aligned[i] or close[i] <= S4_aligned[i]) and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price at or above R3/R4 + daily downtrend + volume
            elif (close[i] >= R3_aligned[i] or close[i] >= R4_aligned[i]) and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price reaches midpoint (mean reversion complete) or trend changes
            midpoint = (S3_aligned[i] + R3_aligned[i]) / 2
            if close[i] >= midpoint or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price reaches midpoint or trend changes
            midpoint = (S3_aligned[i] + R3_aligned[i]) / 2
            if close[i] <= midpoint or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals