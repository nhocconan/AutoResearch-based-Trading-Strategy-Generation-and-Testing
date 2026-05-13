#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike > 1.8x average.
# Long when price closes above R1 with 1d EMA34 uptrend (close > EMA34) and volume > 1.8x 20-bar average volume.
# Short when price closes below S1 with 1d EMA34 downtrend (close < EMA34) and volume > 1.8x average.
# Exit when price reverses and closes below/above the opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 12h timeframe.
# Tighter Camarilla levels (R1/S1) and moderate volume filter (1.8x) reduce overtrading while capturing strong intraday moves.
# 1d EMA34 ensures we only trade in the direction of the higher timeframe trend, improving win rate in both bull and bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Camarilla R1 and S1 levels (tighter breakout levels)
    camarilla_range = high_prev - low_prev
    r1 = close_prev + 1.1 * camarilla_range / 4
    s1 = close_prev - 1.1 * camarilla_range / 4
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 1d EMA34 to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above R1 with 1d EMA34 uptrend and volume spike > 1.8x
            if (close[i] > r1[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with 1d EMA34 downtrend and volume spike > 1.8x
            elif (close[i] < s1[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversal signal)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversal signal)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals