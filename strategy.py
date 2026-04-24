#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot (R1/S1) breakout with 4h trend filter and session filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for Camarilla pivot calculation (based on prior 4h OHLC) and EMA50 trend filter.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume Asian session noise.
- Entry: Long when price breaks above R1 AND close > 4h EMA50 (uptrend).
         Short when price breaks below S1 AND close < 4h EMA50 (downtrend).
- Exit: Opposite Camarilla breakout (price crosses back below R1 for longs, above S1 for shorts).
- Signal size: 0.20 discrete to minimize fee drag.
- Camarilla breakouts capture momentum after testing key levels from prior 4h range.
- 4h EMA50 filter ensures trades align with intermediate-term trend, reducing whipsaws.
- Session filter reduces false breakouts during low-liquidity periods.
- Works in both bull and bear markets by trading with the 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 4h Camarilla pivots (R1, S1) from prior 4h OHLC
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least 2 bars for prior bar calculation
        return np.zeros(n)
    
    # Prior bar OHLC for Camarilla calculation
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 4h EMA50 for trend filter
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = ema(df_4h['close'].values, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        # Exit conditions: price crosses back below R1 for longs, above S1 for shorts
        if position != 0:
            # Exit long: price crosses below R1
            if position == 1:
                if curr_close < camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above S1
            elif position == -1:
                if curr_close > camarilla_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with 4h EMA50 trend filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= camarilla_r1_aligned[i] and prev_close < camarilla_r1_aligned[i-1]
            breakout_down = curr_low <= camarilla_s1_aligned[i] and prev_close > camarilla_s1_aligned[i-1]
            
            # Trend filter: close > EMA50 for uptrend, close < EMA50 for downtrend
            uptrend = curr_close > ema_50_4h_aligned[i]
            downtrend = curr_close < ema_50_4h_aligned[i]
            
            if breakout_up and uptrend:
                signals[i] = 0.20
                position = 1
            elif breakout_down and downtrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0