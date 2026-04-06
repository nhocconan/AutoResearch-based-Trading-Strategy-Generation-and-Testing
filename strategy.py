#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with 1d Trend Filter and Volume Confirmation
Hypothesis: Price often reverses at weekly pivot support/resistance levels (R1, S1, R2, S2) during ranging markets.
Entries are taken when price rejects these levels with volume confirmation, filtered by 1d trend direction (price vs EMA50).
Stops are placed beyond the pivot level. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get weekly pivot data from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll use rolling window of 5 days (1 week) to get weekly OHLC
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    for i in range(5, len(high_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
        weekly_close[i] = close_1d[i-1]  # Previous day's close
    
    # Calculate pivot points and support/resistance levels
    pp = np.full(len(high_1d), np.nan)  # Pivot Point
    r1 = np.full(len(high_1d), np.nan)  # Resistance 1
    s1 = np.full(len(high_1d), np.nan)  # Support 1
    r2 = np.full(len(high_1d), np.nan)  # Resistance 2
    s2 = np.full(len(high_1d), np.nan)  # Support 2
    
    for i in range(5, len(high_1d)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            pp[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            r1[i] = 2 * pp[i] - weekly_low[i]
            s1[i] = 2 * pp[i] - weekly_high[i]
            r2[i] = pp[i] + (weekly_high[i] - weekly_low[i])
            s2[i] = pp[i] - (weekly_high[i] - weekly_low[i])
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d trend filter: price vs EMA50
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    # Trend: 1 if close > EMA50 (bullish), -1 if close < EMA50 (bearish)
    trend_1d = np.where(close_1d > ema_50, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.8x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or \
           np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches R1 or R2, or trend turns bearish
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] >= r1_aligned[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches S1 or S2, or trend turns bullish
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] <= s1_aligned[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for reversal entries at pivot levels
            # Long: price rejects S1 or S2 with volume in bullish 1d trend
            if ((abs(close[i] - s1_aligned[i]) < 0.5 * atr[i] and close[i] > s1_aligned[i]) or
                (abs(close[i] - s2_aligned[i]) < 0.5 * atr[i] and close[i] > s2_aligned[i])) and \
               trend_1d_aligned[i] == 1 and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price rejects R1 or R2 with volume in bearish 1d trend
            elif ((abs(close[i] - r1_aligned[i]) < 0.5 * atr[i] and close[i] < r1_aligned[i]) or
                  (abs(close[i] - r2_aligned[i]) < 0.5 * atr[i] and close[i] < r2_aligned[i])) and \
                 trend_1d_aligned[i] == -1 and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals