#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter
Hypothesis: On 12h timeframe, breakouts above daily R1 or below daily S1 with volume confirmation and aligned 1d trend (EMA50) yield high-probability trades. Uses tight entry criteria to limit trades (12-37/year) and avoid fee drift. Works in bull/bear markets by only taking breakouts in direction of daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1, close  # pivot not used directly

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily Camarilla levels from previous day's OHLC
        # Need to find index of previous completed 1d bar
        # Since we're on 12h timeframe, we use the prior day's data directly
        prev_day_idx = len(df_1d) - 1  # This is approximate; better to use actual alignment
        # Instead, we calculate pivots using the prior completed day's data
        # We'll use a rolling window approach on the daily data
        
        # Get current 12h bar's timestamp to find corresponding prior day
        # Simpler: use the daily data up to the current point
        # We need the prior completed day's OHLC
        # Since we aligned the EMA, we can use the same logic for pivots
        # Extract daily OHLC series
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d_arr = df_1d['close'].values
        
        # Align these to 12h timeframe
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_arr)
        
        # Use prior bar's aligned daily values (previous completed day)
        prev_high = high_1d_aligned[i-1]
        prev_low = low_1d_aligned[i-1]
        prev_close = close_1d_aligned[i-1]
        
        r1, s1, _ = calculate_camarilla(prev_high, prev_low, prev_close)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        trend_long = price > ema_50_1d_aligned[i]
        trend_short = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + uptrend
            if price > r1 and volume_ok and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + downtrend
            elif price < s1 and volume_ok and trend_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns bearish
            if price < s1 or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns bullish
            if price > r1 or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0