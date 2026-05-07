#!/usr/bin/env python3
# 4h_AroonOscillator_Trend_Strength_1dFilter
# Hypothesis: Aroon Oscillator (25-period) identifies strong trends with clear strength signals.
# Values near +100 indicate strong uptrends, near -100 strong downtrends. Uses 1d ADX as regime filter
# to avoid ranging markets. Works in both bull and bear markets by capturing trend strength rather
# than direction alone. Position size 0.25 balances risk and return.

name = "4h_AroonOscillator_Trend_Strength_1dFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Aroon Oscillator on 4h (25-period)
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period, n):
        # Periods since highest high
        highest_high_idx = np.argmax(high[i-aroon_period+1:i+1]) + (i - aroon_period + 1)
        periods_since_high = i - highest_high_idx
        aroon_up[i] = ((aroon_period - periods_since_high) / aroon_period) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low[i-aroon_period+1:i+1]) + (i - aroon_period + 1)
        periods_since_low = i - lowest_low_idx
        aroon_down[i] = ((aroon_period - periods_since_low) / aroon_period) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(aroon_period, 40)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(aroon_osc[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: ADX > 25 indicates trending market
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: strong uptrend (Aroon Osc > 50)
            if aroon_osc[i] > 50 and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (Aroon Osc < -50)
            elif aroon_osc[i] < -50 and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakening or reversal
            if aroon_osc[i] < 0 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakening or reversal
            if aroon_osc[i] > 0 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals