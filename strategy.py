#!/usr/bin/env python3
"""
6h Elder Ray Power + ADX Trend Filter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures
buying/selling pressure relative to trend. Combined with ADX > 25 to ensure we
only trade in trending markets, avoiding whipsaws in ranges. Works in both bull
and bear markets by capturing strong directional moves with clear institutional
participation. Low-frequency design targets 15-25 trades/year to minimize fee drag.
"""
name = "6h_ElderRay_ADX_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === EMA13 for Elder Ray ===
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray Power ===
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === ADX (14) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 1d Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (buying pressure) + ADX > 25 (strong trend) + price above 1d EMA50 (uptrend)
            if (bull_power[i] > 0 and 
                adx[i] > 25 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (selling pressure) + ADX > 25 (strong trend) + price below 1d EMA50 (downtrend)
            elif (bear_power[i] > 0 and 
                  adx[i] > 25 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 OR ADX < 20 (trend weakening) OR price below 1d EMA50
            if (bull_power[i] <= 0 or 
                adx[i] < 20 or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 OR ADX < 20 (trend weakening) OR price above 1d EMA50
            if (bear_power[i] <= 0 or 
                adx[i] < 20 or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals