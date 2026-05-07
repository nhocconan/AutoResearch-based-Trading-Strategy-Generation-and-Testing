#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Elder Ray indicators (13-period EMA as base)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Weekly EMA for trend filter
    ema_13_1w = pd.Series(df_1w['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Volume filter: 1.5x average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1w_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish conditions: bull power > 0, weekly uptrend, volume spike
        bullish = (bull_power[i] > 0) and (ema_13_1w_aligned[i] > ema_13_1w_aligned[i-1]) and (volume[i] > vol_ma_20[i] * 1.5)
        # Bearish conditions: bear power < 0, weekly downtrend, volume spike
        bearish = (bear_power[i] < 0) and (ema_13_1w_aligned[i] < ema_13_1w_aligned[i-1]) and (volume[i] > vol_ma_20[i] * 1.5)
        
        if position == 0:
            if bullish:
                signals[i] = 0.25
                position = 1
            elif bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bull power turns negative or weekly trend changes
            if bull_power[i] <= 0 or ema_13_1w_aligned[i] <= ema_13_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bear power turns positive or weekly trend changes
            if bear_power[i] >= 0 or ema_13_1w_aligned[i] >= ema_13_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation
# - Bull Power = High - EMA13 measures bullish strength
# - Bear Power = Low - EMA13 measures bearish strength
# - Long when Bull Power > 0 (bulls in control) + weekly uptrend + volume confirmation
# - Short when Bear Power < 0 (bears in control) + weekly downtrend + volume confirmation
# - Weekly EMA13 trend filter ensures alignment with higher timeframe trend
# - Works in both bull (buy strength in uptrend) and bear (sell weakness in downtrend)
# - Volume confirmation (1.5x average) reduces false signals
# - Exit when power turns negative/positive or weekly trend changes
# - Position size 0.25 targets ~15-35 trades/year to stay within limits
# - Novel combination: Elder Ray (13/13 EMA) + weekly trend + volume filter not recently tried
# - Aims for 60-140 total trades over 4 years (15-35/year) to stay within limits