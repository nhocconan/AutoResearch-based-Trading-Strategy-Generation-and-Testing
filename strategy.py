#!/usr/bin/env python3
name = "6h_ElderRay_RayPlus_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (on 1d)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Weekly trend: EMA21 on weekly close
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume spike: 12-period average (3 days of 6h bars)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 21, 12)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0 (market in balance), weekly uptrend, volume spike
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and ema21_1w_aligned[i] > ema21_1w_aligned[i-1] and volume[i] > vol_ma_12[i] * 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0 (market in balance), weekly downtrend, volume spike
            elif bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and ema21_1w_aligned[i] < ema21_1w_aligned[i-1] and volume[i] > vol_ma_12[i] * 1.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or weekly trend breaks
            if bull_power_aligned[i] <= 0 or ema21_1w_aligned[i] < ema21_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns negative or weekly trend breaks
            if bear_power_aligned[i] <= 0 or ema21_1w_aligned[i] > ema21_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Elder Ray with weekly trend and volume confirmation
# - Elder Ray measures bull/bear power relative to EMA13 (1d)
# - Long when Bull Power > 0 AND Bear Power < 0 (balanced bullish) + weekly uptrend + volume spike
# - Short when Bear Power > 0 AND Bull Power < 0 (balanced bearish) + weekly downtrend + volume spike
# - Volume confirmation (1.8x average) filters false signals
# - Exits when power shifts or weekly trend breaks
# - Works in bull markets (buy balanced strength) and bear (sell balanced weakness)
# - Position size 0.25 targets 15-35 trades/year, avoiding fee drag
# - Weekly trend filter ensures alignment with higher timeframe momentum