#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA(50) trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 with price > 1w EMA50 (bullish trend) and volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S4 with price < 1w EMA50 (bearish trend) and volume > 1.8x average.
# Exit when price reverses and closes below/above the Camarilla pivot point.
# Uses tighter breakout levels (R4/S4) and higher timeframe trend (weekly) to reduce false signals.
# Volume confirmation lowered to 1.8x to avoid excessive whipsaws while maintaining breakout validation.
# Discrete position sizing 0.25 targets 50-150 total trades over 4 years on 12h timeframe.
# Weekly EMA filter ensures we trade with the dominant long-term trend, improving performance in both bull and bear markets.

name = "12h_Camarilla_R4_S4_1wEMA50_Trend_VolumeConfirm"
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
    
    lookback = 20  # for volume average
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    if len(close_1w) < 50:
        ema_50_1w = np.full(len(close_1w), np.nan)
    else:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1, Pivot = (H+L+C)/3
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    camarilla_pivot = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        camarilla_r4[i] = c + (h - l) * 1.1
        camarilla_s4[i] = c - (h - l) * 1.1
        camarilla_pivot[i] = (h + l + c) / 3.0
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4 with bullish 1w EMA trend and volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S4 with bearish 1w EMA trend and volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals