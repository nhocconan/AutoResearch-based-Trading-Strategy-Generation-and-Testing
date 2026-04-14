#!/usr/bin/env python3
"""
1d Bollinger Band Breakout + 1W Trend + Volume Spike
Long when price closes above upper Bollinger Band, weekly close > weekly open, and volume > 1.5x average.
Short when price closes below lower Bollinger Band, weekly close < weekly open, and volume > 1.5x average.
Exit when price reverses back through the Bollinger middle band.
Designed for low turnover: ~5-15 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate 20-period Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
    
    upper_band = sma + bb_std * std_dev
    lower_band = sma - bb_std * std_dev
    middle_band = sma
    
    # Volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / vol_count
            vol_sum -= volume[i - 19]
            vol_count -= 1
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Create arrays for alignment
    weekly_bullish_arr = weekly_bullish.astype(float)
    weekly_bearish_arr = weekly_bearish.astype(float)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(bb_period - 1, n):
        # Skip if any indicator not ready
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get aligned weekly trend values
        weekly_bull = align_htf_to_ltf(prices, df_1w, weekly_bullish_arr)[i]
        weekly_bear = align_htf_to_ltf(prices, df_1w, weekly_bearish_arr)[i]
        
        if np.isnan(weekly_bull) or np.isnan(weekly_bear):
            continue
        
        if position == 0:
            # Long: Close above upper Bollinger Band, volume spike, weekly bullish
            if close[i] > upper_band[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bull > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Close below lower Bollinger Band, volume spike, weekly bearish
            elif close[i] < lower_band[i] and volume[i] > vol_ma[i] * 1.5 and weekly_bear > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price closes below middle Bollinger Band
            if close[i] < middle_band[i] and close[i-1] >= middle_band[i-1]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price closes above middle Bollinger Band
            if close[i] > middle_band[i] and close[i-1] <= middle_band[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0