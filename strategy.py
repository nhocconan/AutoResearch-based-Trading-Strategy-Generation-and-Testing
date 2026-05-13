#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA(50) trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 with price > 12h EMA50 (bullish trend) and volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S1 with price < 12h EMA50 (bearish trend) and volume > 1.8x average.
# Exit when price reverses and closes below/above the Camarilla pivot point (mean reversion exit).
# Uses discrete position sizing 0.28. Target: 75-200 total trades over 4 years on 4h timeframe.
# EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Camarilla exit provides clear, objective stop.
# Camarilla pivot levels work well in both trending and ranging markets, providing clear breakout levels.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    lookback = 20  # for volume average
    
    # Calculate Camarilla pivot levels (based on previous bar)
    pivot = (high[:-1] + low[:-1] + close[:-1]) / 3.0
    range_ = high[:-1] - low[:-1]
    r1 = pivot + range_ * 1.1 / 12
    s1 = pivot - range_ * 1.1 / 12
    # Shift to align with current bar (previous bar's levels)
    camarilla_r1 = np.concatenate([np.array([np.nan]), r1])
    camarilla_s1 = np.concatenate([np.array([np.nan]), s1])
    camarilla_pivot = np.concatenate([np.array([np.nan]), pivot])
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close
    if len(close_12h) < 50:
        ema_50_12h = np.full(len(close_12h), np.nan)
    else:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 with bullish 12h EMA trend and volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.28
                position = 1
            # SHORT: Price breaks below Camarilla S1 with bearish 12h EMA trend and volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals