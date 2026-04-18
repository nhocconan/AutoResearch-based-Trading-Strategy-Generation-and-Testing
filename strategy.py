#!/usr/bin/env python3
"""
1h Momentum Pullback with 4h Trend Filter
Hypothesis: In strong trends (identified by 4h EMA alignment and ADX), pullbacks to the 1h EMA offer high-probability entry points. Volume confirms institutional interest. Works in both bull and bear markets by following the 4h trend direction. Low trade frequency due to strict multi-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) and EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values with Wilder smoothing
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = wilders_smooth(tr, 14)
    plus_di_4h = 100 * wilders_smooth(plus_dm, 14) / np.where(atr_4h != 0, atr_4h, 1)
    minus_di_4h = 100 * wilders_smooth(minus_dm, 14) / np.where(atr_4h != 0, atr_4h, 1)
    dx_4h = np.where((plus_di_4h + minus_di_4h) != 0, 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h), 0)
    adx_4h = wilders_smooth(dx_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1h EMA(34) for pullback entries
    ema34_1h = pd.Series(close).ewm(span=34, adjust=False).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma_20[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]) or np.isnan(ema34_1h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend direction
        uptrend_4h = ema20_4h_aligned[i] > ema50_4h_aligned[i]
        downtrend_4h = ema20_4h_aligned[i] < ema50_4h_aligned[i]
        strong_trend = adx_4h_aligned[i] > 25
        
        vol_ok = volume[i] > vol_threshold[i]
        
        if position == 0:
            # Enter long: 4h uptrend + strong trend + pullback to EMA34 + volume
            if (uptrend_4h and strong_trend and 
                close[i] <= ema34_1h[i] * 1.005 and  # Allow small overshoot
                close[i] >= ema34_1h[i] * 0.995 and
                vol_ok):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + strong trend + pullback to EMA34 + volume
            elif (downtrend_4h and strong_trend and 
                  close[i] >= ema34_1h[i] * 0.995 and  # Allow small overshoot
                  close[i] <= ema34_1h[i] * 1.005 and
                  vol_ok):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend weakens or price moves too far from EMA
            if not (uptrend_4h and strong_trend) or close[i] > ema34_1h[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend weakens or price moves too far from EMA
            if not (downtrend_4h and strong_trend) or close[i] < ema34_1h[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Momentum_Pullback_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0