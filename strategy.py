#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal strategy with daily trend filter
# Uses daily Camarilla levels (R3/S3 for reversal, R4/S4 for breakout) from previous day
# Long when price crosses above S3 with close > S3 and daily close > daily open (uptrend)
# Short when price crosses below R3 with close < R3 and daily close < daily open (downtrend)
# Exit when price crosses opposite Camarilla level (S4 for long, R4 for short) or opposite S3/R3
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous day's high
    prev_low = df_1d['low'].shift(1).values    # previous day's low
    prev_close = df_1d['close'].shift(1).values # previous day's close
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    s3 = pivot - (range_hl * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    r4 = pivot + (range_hl * 1.1)        # R4 = pivot + 1.1*(H-L)
    s4 = pivot - (range_hl * 1.1)        # S4 = pivot - 1.1*(H-L)
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels for current day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily trend filter: today's close > today's open for uptrend, < for downtrend
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_uptrend = daily_close > daily_open  # today's close > today's open
    daily_downtrend = daily_close < daily_open  # today's close < today's open
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(1, n):  # Start from 1 to avoid issues with shift
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below S4 or crosses below S3 (contrarian exit)
            elif close[i] < s4_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above R4 or crosses above R3 (contrarian exit)
            elif close[i] > r4_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla reversal with daily trend filter
            # Long: price crosses above S3 with daily uptrend
            if (close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and 
                daily_uptrend_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below R3 with daily downtrend
            elif (close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and 
                  daily_downtrend_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals