#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: On 4-hour timeframe, use Camarilla pivot R1/S1 breakouts in the direction of 1-day trend (via EMA34) with volume confirmation. Daily trend filter avoids counter-trend trades, while Camarilla levels provide precise entry/exit points. Volume surge confirms institutional participation. Designed for moderate trade frequency (~20-50/year) to balance opportunity and fee decay in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily trend: bullish when price > EMA34
    daily_uptrend = close > ema34_daily_aligned
    daily_downtrend = close < ema34_daily_aligned
    
    # Calculate previous day's Camarilla pivot levels (R1, S1)
    # Using prior day's high, low, close
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > r1_aligned[i] and daily_uptrend[i] and volume_surge[i]
        short_entry = close[i] < s1_aligned[i] and daily_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level with volume surge
        long_exit = close[i] < s1_aligned[i] and volume_surge[i]
        short_exit = close[i] > r1_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0