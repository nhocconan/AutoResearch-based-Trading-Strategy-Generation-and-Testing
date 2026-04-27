#!/usr/bin/env python3
"""
1d_WeeklyATRBreakout_1wTrend_Filter
Hypothesis: Daily breakout of weekly ATR-based channel with weekly trend filter.
In bull markets: buy when price exceeds weekly ATR upper band (close + 2*ATR).
In bear markets: sell when price breaks below weekly ATR lower band (close - 2*ATR).
Weekly trend filter (price vs weekly EMA20) prevents counter-trend entries.
Target: 15-30 trades per year to minimize fee drag while capturing major moves.
Works in both bull and bear by following weekly trend direction.
"""

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
    
    # Get weekly data for ATR and EMA calculations
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    tr_weekly = np.maximum(high_weekly - low_weekly,
                          np.maximum(np.abs(high_weekly - np.roll(close_weekly, 1)),
                                   np.abs(low_weekly - np.roll(close_weekly, 1))))
    tr_weekly[0] = high_weekly[0] - low_weekly[0]
    
    atr_weekly = np.full(len(close_weekly), np.nan)
    atr_period = 14
    for i in range(atr_period, len(close_weekly)):
        atr_weekly[i] = np.mean(tr_weekly[i-atr_period+1:i+1])
    
    # Calculate weekly EMA20 for trend filter
    ema_period = 20
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period-1] = np.mean(close_weekly[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * multiplier) + (ema_weekly[i-1] * (1 - multiplier))
    
    # Calculate weekly ATR-based channels
    upper_band = close_weekly + 2.0 * atr_weekly
    lower_band = close_weekly - 2.0 * atr_weekly
    
    # Align weekly indicators to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_band)
    ema_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = max(atr_period, ema_period)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Weekly trend filter
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        if position == 0:
            # Enter long when price breaks above weekly upper band in uptrend
            if uptrend and price > upper_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short when price breaks below weekly lower band in downtrend
            elif downtrend and price < lower_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price crosses below weekly EMA (trend change)
            if not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Exit short when price crosses above weekly EMA (trend change)
            if not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_WeeklyATRBreakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0