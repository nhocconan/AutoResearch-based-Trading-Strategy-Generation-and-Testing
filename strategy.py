#!/usr/bin/env python3
# 4H_1D_4H_EMA_Cross_With_1D_Trend
# Hypothesis: Use 4h EMA cross for entry timing, filtered by 1d EMA trend and volume confirmation.
# Only trade in direction of 1d trend to avoid counter-trend whipsaws. 4h EMA cross provides
# timely entries while 1d trend filter improves win rate. Volume confirmation ensures
# momentum behind moves. Designed for fewer trades (<50/year) to minimize fee drag.

name = "4H_1D_4H_EMA_Cross_With_1D_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: EMA(50) - slower for stronger trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema_50_1d
    
    # Volume confirmation: current volume > 1.8x 30-period average (stricter)
    volume_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (volume_avg * 1.8)
    
    # 4h EMA cross: fast EMA(12) and slow EMA(26)
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Bullish cross: fast crosses above slow
    bullish_cross = (ema_12 > ema_26) & (np.roll(ema_12, 1) <= np.roll(ema_26, 1))
    # Bearish cross: fast crosses below slow
    bearish_cross = (ema_12 < ema_26) & (np.roll(ema_12, 1) >= np.roll(ema_26, 1))
    # Handle first element
    bullish_cross[0] = False
    bearish_cross[0] = False
    
    # Align 1d indicators to 4h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish EMA cross + 1d uptrend + volume confirmation
            if bullish_cross[i] and trend_up_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish EMA cross + 1d downtrend + volume confirmation
            elif bearish_cross[i] and not trend_up_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish EMA cross or 1d trend turns down
            if bearish_cross[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish EMA cross or 1d trend turns up
            if bullish_cross[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals