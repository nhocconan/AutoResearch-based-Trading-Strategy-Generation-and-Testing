#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA(50) trend filter and 12h volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) with 1d EMA(50) bullish (close > EMA) and 12h volume > 1.8x 20-period average.
# Short when price < Alligator Jaw with 1d EMA(50) bearish (close < EMA) and 12h volume > 1.8x 20-period average.
# Exit on opposite Alligator Teeth (8-period SMMA) for longs, Alligator Lips (5-period SMMA) for shorts.
# Uses Williams Alligator (SMMA-based) for trend identification, which whipsaws less than EMA/HMA in ranging markets.
# 1d EMA(50) ensures multi-timeframe trend alignment. Volume filter confirms institutional participation.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe.
# Works in bull/bear: 1d EMA filters counter-trend signals, Alligator adapts to volatility, volume avoids false breakouts.

name = "12h_WilliamsAlligator_1dEMA50_12hVolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) as used in Williams Alligator"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return result
    # First value is SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_PRICE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 12h volume confirmation: > 1.8x 20-period average (tight filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Jaw + 1d EMA bullish + volume confirmation
            if (close[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Jaw + 1d EMA bearish + volume confirmation
            elif (close[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < Teeth (8-period SMMA)
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > Lips (5-period SMMA)
            if close[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals