#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation. 
# Uses 4h EMA50 for trend direction, 1h Camarilla levels for entry timing, and volume spike for confirmation.
# Designed for 1h timeframe with tight entries to avoid overtrading. Target: 60-150 trades over 4 years.
# Works in both bull/bear regimes: trend filter ensures alignment with higher timeframe, volume confirms institutional interest.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
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
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels (R1, S1)
    camarilla_r1 = close + (high - low) * 1.1 / 12
    camarilla_s1 = close - (high - low) * 1.1 / 12
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, price > 4h EMA50, volume spike (>1.5x avg)
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1, price < 4h EMA50, volume spike (>1.5x avg)
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla S1 (mean reversion) OR closes below 4h EMA50
            if (close[i] < camarilla_s1[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla R1 (mean reversion) OR closes above 4h EMA50
            if (close[i] > camarilla_r1[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals