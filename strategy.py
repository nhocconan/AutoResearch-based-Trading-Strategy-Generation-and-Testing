#!/usr/bin/env python3
"""
6h Elder Ray Index + Weekly Trend Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) identifies institutional buying/selling pressure.
Combined with weekly trend (price vs 200 EMA) to filter direction. Works in bull (buy on bull power)
and bear (sell on bear power). Target: 80-150 total trades over 4 years (20-38/year).
Uses weekly trend filter to avoid counter-trend trades. Volume confirmation reduces false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14435_6h_elder_ray_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly 200 EMA for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Smooth the power values (13-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, min_periods=13, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(13, 200) + 10
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: bear power turns positive (selling pressure) OR stoploss
            if (bear_power_smooth[i] > 0 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns negative (buying pressure) OR stoploss
            if (bull_power_smooth[i] < 0 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals + weekly trend + volume
            long_setup = (bull_power_smooth[i] > 0 and bear_power_smooth[i] < 0 and
                         weekly_uptrend and vol_filter[i])
            short_setup = (bear_power_smooth[i] > 0 and bull_power_smooth[i] < 0 and
                          weekly_downtrend and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals