#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume spike > 1.8x average.
# Long when price closes above R4 with 1w EMA50 uptrend (price > EMA50) and volume > 1.8x 20-bar average volume.
# Short when price closes below S4 with 1w EMA50 downtrend (price < EMA50) and volume > 1.8x average.
# Exit when price reverses and closes below/above the opposite Camarilla level (S4 for longs, R4 for shorts).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 12h timeframe.
# Higher timeframe (12h) reduces trade frequency, minimizing fee drag. Volume confirmation and 1w trend filter
# ensure trades are taken only in strong momentum conditions, improving win rate and reducing false breakouts.
# 1w EMA50 provides a smooth trend filter that adapts to both bull and bear markets.

name = "12h_Camarilla_R4_S4_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate Camarilla levels from previous day (approx using 2x 12h bars)
    lookback = 2  # 2 * 12h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R4 and S4 levels
    camarilla_range = high_prev - low_prev
    r4 = close_prev + 1.5 * camarilla_range / 2
    s4 = close_prev - 1.5 * camarilla_range / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA50 to 12h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above R4 with 1w EMA50 uptrend and volume spike > 1.8x
            if (close[i] > r4[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S4 with 1w EMA50 downtrend and volume spike > 1.8x
            elif (close[i] < s4[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S4 (reversal signal)
            if close[i] < s4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R4 (reversal signal)
            if close[i] > r4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals