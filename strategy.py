#!/usr/bin/env python3
"""
6H Weekly Pivot Breakout with Volume Confirmation and RSI Filter
Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance. Breakouts beyond these levels with volume confirmation and RSI momentum capture strong directional moves. Weekly context avoids whipsaw, suitable for both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_rsi_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly OHLC for pivot calculation
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points and support/resistance levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    weekly_r4 = weekly_r3 + (weekly_high - weekly_low)
    weekly_s4 = weekly_s3 - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14-period) on 6h
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    rsi_period = 14
    avg_gain = wilder_smooth(gain, rsi_period)
    avg_loss = wilder_smooth(loss, rsi_period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20 + rsi_period, 20)  # For RSI and volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to weekly pivot or RSI extreme reversal
        if position == 1:  # long position
            # Exit: price crosses below weekly pivot OR RSI > 70 (overbought)
            if close[i] < pivot_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above weekly pivot OR RSI < 30 (oversold)
            if close[i] > pivot_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout beyond R4/S4 + volume + RSI momentum
            bull_breakout = close[i] > r4_aligned[i]
            bear_breakout = close[i] < s4_aligned[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            rsi_momentum = (rsi[i] > 50 and rsi[i] < 70) for long, (rsi[i] < 50 and rsi[i] > 30) for short
            
            if bull_breakout and volume_filter and (rsi[i] > 50 and rsi[i] < 70):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and (rsi[i] < 50 and rsi[i] > 30):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals