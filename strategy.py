#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with price > 1d EMA50 (bullish trend) and volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S3 with price < 1d EMA50 (bearish trend) and volume > 1.5x average.
# Exit when price reverses and closes below/above the Camarilla pivot point (PP).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 12h timeframe.
# EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Camarilla PP exit provides clear, objective stop.
# Works in both bull and bear markets by following the 1d trend.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Calculate Camarilla pivot levels (based on previous bar)
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3
    r3 = pp + (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 2
    s3 = pp - (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    if len(close_1d) < 50:
        ema_50_1d = np.full(len(close_1d), np.nan)
    else:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume avg
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with bullish 1d EMA trend and volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with bearish 1d EMA trend and volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below PP (reversal signal)
            if close[i] < pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above PP (reversal signal)
            if close[i] > pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals