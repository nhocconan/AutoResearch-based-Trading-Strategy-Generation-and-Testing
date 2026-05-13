#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 and close > 1d EMA50 with volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S4 and close < 1d EMA50 with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 50-100 total trades over 4 years on 4h timeframe.
# Camarilla R4/S4 are stronger intraday levels than R1/S1/R3/S3, reducing false breakouts and overtrading.
# 1d EMA50 ensures higher timeframe trend alignment; volume confirmation filters weak momentum.
# This variant targets fewer, higher-quality trades to avoid fee drag while maintaining edge in both bull and bear markets.

name = "4h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_Volume"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla R4 = close_prev + (high_prev - low_prev) * 1.1/2
    # Camarilla S4 = close_prev - (high_prev - low_prev) * 1.1/2
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    camarilla_r4 = close_prev + (high_prev - low_prev) * 1.1 / 2
    camarilla_s4 = close_prev - (high_prev - low_prev) * 1.1 / 2
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate average volume for confirmation (20-period)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4, close > 1d EMA50, volume confirmation
            if (high[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S4, close < 1d EMA50, volume confirmation
            elif (low[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S4 OR volume drops below average
            if (low[i] < camarilla_s4_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R4 OR volume drops below average
            if (high[i] > camarilla_r4_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals