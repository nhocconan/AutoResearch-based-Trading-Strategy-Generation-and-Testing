#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) and 1-day EMA50 is rising with volume spike.
Short when price breaks below lower Donchian(20) and 1-day EMA50 is falling with volume spike.
Exit when price crosses the opposite Donchian band or 1-day EMA50 reverses direction.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-day close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian and 1-day EMA50 rising with volume spike
            if close[i] > upper[i] and ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian and 1-day EMA50 falling with volume spike
            elif close[i] < lower[i] and ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian band or 1-day EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below lower Donchian or 1-day EMA50 turns down
                if close[i] < lower[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above upper Donchian or 1-day EMA50 turns up
                if close[i] > upper[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0