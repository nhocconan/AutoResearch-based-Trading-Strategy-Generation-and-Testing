#!/usr/bin/env python3
"""
6h Heikin-Ashi Smoothed Breakout with Weekly Trend Filter
Hypothesis: Heikin-Ashi smoothing filters noise, breakouts capture momentum, weekly trend filter ensures directional bias. Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee decay in choppy 2025 market.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ha_breakout_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Heikin-Ashi
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # 20-period ATR for volatility and stops
    atr = np.full(n, np.nan)
    if n >= 21:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 19 + atr[i-1]) / 20
    
    # Weekly trend filter (using weekly close vs weekly open)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) > 0:
        weekly_close = df_weekly['close'].values
        weekly_open = df_weekly['open'].values
        weekly_up = weekly_close > weekly_open
        weekly_trend = align_htf_to_ltf(prices, df_weekly, weekly_up.astype(float))
    else:
        weekly_trend = np.ones(n)  # default to long bias if no weekly data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For HA and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ha_close[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend (already shifted by align_htf_to_ltf)
        trend_bull = weekly_trend[i] > 0.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: HA close below prior HA open (trend change)
            # Stoploss: price drops 2.5*ATR below entry
            if (ha_close[i] < ha_open[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: HA close above prior HA open (trend change)
            # Stoploss: price rises 2.5*ATR above entry
            if (ha_close[i] > ha_open[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: HA breakout + weekly trend filter
            bull_breakout = ha_close[i] > ha_high[i-1]
            bear_breakout = ha_close[i] < ha_low[i-1]
            
            if bull_breakout and trend_bull:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and not trend_bull:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals