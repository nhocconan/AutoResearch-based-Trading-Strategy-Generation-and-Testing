#!/usr/bin/env python3
"""
4h_1d1w_Pivot_Breakout_Volume_Strict_V1
Hypothesis: On 4h timeframe, buy when price breaks above daily Camarilla R1 with volume spike (>2x avg volume) during strong weekly trend (ADX > 25), sell when breaks below daily S1. Exit on opposite breakout or trend weakening (ADX < 20). Uses strict volume confirmation and weekly ADX to filter false breakouts. Target: 15-25 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smooth_wilder(tr, period)
    plus_di = 100 * smooth_wilder(plus_dm, period) / atr
    minus_di = 100 * smooth_wilder(minus_dm, period) / atr
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_wilder(dx, period)
    return adx

def calculate_camarilla(high, low, close):
    # Camarilla pivot levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    range_hl = high - low
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Data (HTF for Camarilla levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ADX (14-period)
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Daily Camarilla levels (R1, S1)
    r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation on daily
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === Weekly Data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ADX (14-period)
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_spike = vol_1d_current > 2.0 * vol_ma_1d_aligned[i]
        
        # Trend filters: only trade when both daily and weekly ADX > 25 (strong trend)
        strong_trend_daily = adx_1d_aligned[i] > 25
        strong_trend_weekly = adx_1w_aligned[i] > 25
        strong_trend = strong_trend_daily and strong_trend_weekly
        
        # Exit when either trend weakens (ADX < 20)
        weak_trend_daily = adx_1d_aligned[i] < 20
        weak_trend_weekly = adx_1w_aligned[i] < 20
        weak_trend = weak_trend_daily or weak_trend_weekly
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above daily R1 with volume spike and strong trend
            if close[i] > r1_1d_aligned[i] and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below daily S1 with volume spike and strong trend
            elif close[i] < s1_1d_aligned[i] and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit conditions: trend weakening OR opposite breakout (below S1)
            if weak_trend or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakening OR opposite breakout (above R1)
            if weak_trend or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d1w_Pivot_Breakout_Volume_Strict_V1"
timeframe = "4h"
leverage = 1.0