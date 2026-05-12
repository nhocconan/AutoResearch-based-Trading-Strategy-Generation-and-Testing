#!/usr/bin/env python3
"""
4H_CAMARILLA_R1_S1_BREAKOUT_1D_VOLUME_CONFIRMATION
Hypothesis: Trade breakouts from Camarilla R1 (long) and S1 (short) on 4h timeframe with volume confirmation and 1d EMA50 trend filter. Works in both bull and bear markets by aligning with daily trend while using volatility-based entry levels. Target: 20-40 trades/year.
"""
name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_VOLUME_CONFIRMATION"
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
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rng = ph - pl
        camarilla_r1[i] = pc + 1.1 * rng / 12
        camarilla_s1[i] = pc - 1.1 * rng / 12
    
    # EMA50 for 1d trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 4h volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    # Align all 1d data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        bullish_trend = close[i] > ema50_aligned[i]
        bearish_trend = close[i] < ema50_aligned[i]
        
        if position == 0:
            # LONG: Break above Camarilla R1 with volume spike and bullish trend
            if (high[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                bullish_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 with volume spike and bearish trend
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  bearish_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below Camarilla pivot or trend turns bearish
            camarilla_pivot = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0 if i > 0 and not np.isnan(high_1d[i-1]) else np.nan
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.array([camarilla_pivot]))[0] if i > 0 and not np.isnan(camarilla_pivot) else np.nan
            if (i > 0 and not np.isnan(camarilla_pivot_aligned) and close[i] < camarilla_pivot_aligned) or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above Camarilla pivot or trend turns bullish
            camarilla_pivot = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0 if i > 0 and not np.isnan(high_1d[i-1]) else np.nan
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.array([camarilla_pivot]))[0] if i > 0 and not np.isnan(camarilla_pivot) else np.nan
            if (i > 0 and not np.isnan(camarilla_pivot_aligned) and close[i] > camarilla_pivot_aligned) or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals