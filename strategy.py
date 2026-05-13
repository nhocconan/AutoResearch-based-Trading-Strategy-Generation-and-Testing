#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA trend filter and volume spike confirmation.
# Long when price breaks above R4 with 12h EMA50 > EMA200 (bullish trend) and volume > 1.5x average.
# Short when price breaks below S4 with 12h EMA50 < EMA200 (bearish trend) and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Camarilla R4/S4 levels provide stronger breakout confirmation than R3/S3, reducing false signals.
# 12h EMA crossover filters for intermediate-term trend alignment, improving performance in both bull and bear markets.
# Volume spike confirms institutional participation. Tight entry conditions minimize fee drag.

name = "4h_Camarilla_R4_S4_Breakout_12hEMA_Cross_VolumeSpike_v1"
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
    
    # Calculate Camarilla levels from previous day (approx 6*4h bars)
    lookback = 6  # 6 * 4h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R4 and S4 levels (more extreme than R3/S3)
    # R4 = close_prev + 1.1 * (high_prev - low_prev)
    # S4 = close_prev - 1.1 * (high_prev - low_prev)
    camarilla_range = high_prev - low_prev
    r4 = close_prev + 1.1 * camarilla_range
    s4 = close_prev - 1.1 * camarilla_range
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 12h EMAs to 4h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 200, n):  # Start after sufficient data for EMA200
        # Skip if any required data is NaN
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 with bullish 12h trend (EMA50 > EMA200) and volume spike
            if (close[i] > r4[i] and 
                ema_50_12h_aligned[i] > ema_200_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with bearish 12h trend (EMA50 < EMA200) and volume spike
            elif (close[i] < s4[i] and 
                  ema_50_12h_aligned[i] < ema_200_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 (reversal signal) OR trend turns bearish
            if (close[i] < s4[i]) or (ema_50_12h_aligned[i] < ema_200_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 (reversal signal) OR trend turns bullish
            if (close[i] > r4[i]) or (ema_50_12h_aligned[i] > ema_200_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals