#!/usr/bin/env python3
"""
12h_SuperTrend_With_1d_Trend_Filter
Hypothesis: On 12h timeframe, use Supertrend (ATR=10, multiplier=3) as trend filter, with entries triggered when price closes above/below Supertrend line and volume confirms (>1.5x 20-period average). Daily trend filter (via 8/21 EMA crossover) ensures alignment with higher timeframe trend, reducing counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull/bear markets via dual timeframe trend alignment.
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
    volume = prices['volume'].values
    
    # Calculate ATR for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: no previous close
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 21:
        return np.zeros(n)
    
    # Calculate daily 8 and 21 EMA for trend filter
    close_daily = df_daily['close'].values
    ema8_daily = pd.Series(close_daily).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily EMAs to 12h timeframe
    ema8_daily_aligned = align_htf_to_ltf(prices, df_daily, ema8_daily)
    ema21_daily_aligned = align_htf_to_ltf(prices, df_daily, ema21_daily)
    
    # Daily trend: bullish when EMA8 > EMA21
    daily_uptrend = ema8_daily_aligned > ema21_daily_aligned
    daily_downtrend = ema8_daily_aligned < ema21_daily_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for daily EMA21 and ATR to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or
            np.isnan(ema8_daily_aligned[i]) or np.isnan(ema21_daily_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price close relative to Supertrend + daily trend alignment + volume surge
        long_entry = (close[i] > supertrend[i]) and daily_uptrend[i] and volume_surge[i]
        short_entry = (close[i] < supertrend[i]) and daily_downtrend[i] and volume_surge[i]
        
        # Exit when price crosses Supertrend in opposite direction with volume surge
        long_exit = (close[i] < supertrend[i]) and volume_surge[i]
        short_exit = (close[i] > supertrend[i]) and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_SuperTrend_With_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0