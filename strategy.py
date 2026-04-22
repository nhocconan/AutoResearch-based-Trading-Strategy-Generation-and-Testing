#!/usr/bin/env python3
"""
Hypothesis: 6-hour Keltner Channel breakout with 1-day ADX trend filter and 1-week momentum confirmation.
- Long when price breaks above upper Keltner Channel (EMA20 + 2*ATR10) and ADX(14) > 25 on daily and weekly RSI > 50.
- Short when price breaks below lower Keltner Channel (EMA20 - 2*ATR10) and ADX(14) > 25 on daily and weekly RSI < 50.
- Exit when price crosses back inside the Keltner Channel or ADX drops below 20.
- Designed for low trade frequency by requiring multiple timeframe confirmation and strong trend conditions.
- Works in both bull and bear markets by following the higher timeframe trend.
"""

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
    
    # Keltner Channel components on 6h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr10 = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=10, min_periods=10).mean().values
    kc_upper = ema20 + 2 * atr10
    kc_lower = ema20 - 2 * atr10
    
    # Load daily data for ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # ADX calculation on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load weekly data for momentum confirmation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly RSI
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper KC, ADX > 25, weekly RSI > 50
            if (close[i] > kc_upper[i] and adx_aligned[i] > 25 and rsi_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC, ADX > 25, weekly RSI < 50
            elif (close[i] < kc_lower[i] and adx_aligned[i] > 25 and rsi_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back inside KC or ADX drops below 20
            exit_signal = False
            
            if position == 1:
                # Exit long: price below upper KC or ADX weakens
                if close[i] < kc_upper[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above lower KC or ADX weakens
                if close[i] > kc_lower[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Keltner_ADX_WeeklyRSI"
timeframe = "6h"
leverage = 1.0