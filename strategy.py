#!/usr/bin/env python3
"""
6h_Elder_Ray_Power_Trend_Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) combined with 1d EMA trend filter and volume confirmation works in both bull and bear markets.
Bull Power = High - EMA13, Bear Power = EMA13 - Low.
Go long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 1d uptrend and volume spike.
Go short when Bear Power < 0 and falling, Bull Power < 0 and rising, with 1d downtrend and volume spike.
Exit when power signals weaken or trend reverses. Uses 60-period EMA for power calculation and 20-period volume average.
Target: 15-35 trades/year per symbol.
"""

name = "6h_Elder_Ray_Power_Trend_Filter"
timeframe = "6h"
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
    
    # Elder Ray components: EMA13 for power calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High minus EMA13
    bear_power = low - ema_13   # Low minus EMA13 (negative when low < EMA13)
    
    # Smooth the power signals to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # 60-period EMA for trend context (longer-term filter)
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    uptrend_6h = close > ema_60
    downtrend_6h = close < ema_60
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Get values
        bull = bull_power_smooth[i]
        bear = bear_power_smooth[i]
        uptrend = uptrend_6h[i]
        downtrend = downtrend_6h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: Bull power positive and rising, bear power negative, 6h uptrend, 1d uptrend, volume confirmation
            if bull > 0 and bull > bull_power_smooth[i-1] and bear < 0 and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power negative and falling, bull power positive, 6h downtrend, 1d downtrend, volume confirmation
            elif bear < 0 and bear < bear_power_smooth[i-1] and bull > 0 and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull power turns negative or trend turns down
            if bull <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power turns positive or trend turns up
            if bear >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals