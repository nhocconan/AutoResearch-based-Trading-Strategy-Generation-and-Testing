#!/usr/bin/env python3
"""
1d_1w_Pivot_Breakout_Volume_Strict_V1
Hypothesis: On daily timeframe, buy when price breaks above weekly Camarilla R1 with volume spike (>1.5x avg volume) during strong weekly trend (ADX > 25), sell when breaks below weekly S1. Exit on opposite breakout or trend weakening (ADX < 20). Uses weekly pivot levels for structure, volume confirmation to avoid false breakouts, and weekly ADX to filter chop. Target: 10-20 trades/year for low fee drag.
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
    
    # === Weekly Data (HTF for pivot levels and trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ADX (14-period)
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Weekly Camarilla levels (R1, S1)
    r1_1w, s1_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation on weekly
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current weekly bar's volume for confirmation
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_spike = vol_1w_current > 1.5 * vol_ma_1w_aligned[i]
        
        # Trend filter: only trade when weekly ADX > 25 (strong trend)
        strong_trend = adx_1w_aligned[i] > 25
        
        # Exit when trend weakens (ADX < 20)
        weak_trend = adx_1w_aligned[i] < 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike and strong trend
            if close[i] > r1_1w_aligned[i] and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly S1 with volume spike and strong trend
            elif close[i] < s1_1w_aligned[i] and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit conditions: trend weakening OR opposite breakout (below S1)
            if weak_trend or close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakening OR opposite breakout (above R1)
            if weak_trend or close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Pivot_Breakout_Volume_Strict_V1"
timeframe = "1d"
leverage = 1.0