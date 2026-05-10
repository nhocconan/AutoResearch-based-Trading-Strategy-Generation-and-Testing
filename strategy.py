# 4H_1D_1W_Keltner_Breakout_Trend_Signal
# Hypothesis: Use weekly Keltner Channel breakout with daily trend filter (ADX) to capture strong trends in both bull and bear markets.
# Weekly Keltner provides volatility-adjusted breakout levels, daily ADX filters for trending conditions.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "4H_1D_1W_Keltner_Breakout_Trend_Signal"
timeframe = "4h"
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
    
    # Get weekly data for Keltner Channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Keltner Channel (20, 1.5)
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        multiplier = 2 / (period + 1)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    def atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return ema(tr, period)
    
    keltner_mid = ema(close_1w, 20)
    keltner_atr = atr(high_1w, low_1w, close_1w, 10)
    keltner_upper = keltner_mid + 1.5 * keltner_atr
    keltner_lower = keltner_mid - 1.5 * keltner_atr
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ADX (14)
    def wilders_smoothing(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_data = wilders_smoothing(tr, 14)
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr_data
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr_data
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly Keltner to 4h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Align daily ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        is_trending = adx_val > 25
        
        if position == 0:
            # Enter long: weekly breakout above upper + daily trend
            if close[i] > keltner_upper_aligned[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly breakout below lower + daily trend
            elif close[i] < keltner_lower_aligned[i] and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly break below lower or trend weakens
            if close[i] < keltner_lower_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly break above upper or trend weakens
            if close[i] > keltner_upper_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals