#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Reversal_Volume_Spike
# Hypothesis: Reversal at Camarilla R4/S4 levels with volume spike on 4h chart, filtered by 1-week ADX trend strength.
# Works in both bull and bear markets: 
# - In bull markets, buy S4 bounce in uptrend, sell R4 pullback
# - In bear markets, sell R4 bounce in downtrend, buy S4 pullback
# Uses volume confirmation to avoid low-liquidity whipsaws. Target: 30-60 trades/year.

name = "4h_Camarilla_Pivot_Reversal_Volume_Spike"
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
    
    # 1w data for trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_dm_sum = np.zeros_like(plus_dm)
        minus_dm_sum = np.zeros_like(minus_dm)
        plus_dm_sum[period-1] = np.sum(plus_dm[1:period])
        minus_dm_sum[period-1] = np.sum(minus_dm[1:period])
        
        for i in range(period, len(plus_dm)):
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_sum / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_sum / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # 1d data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.1 * (high - low) / 2
    # S4 = close - 1.1 * (high - low) / 2
    rng = high_1d - low_1d
    r4_1d = close_1d + (1.1 * rng) / 2
    s4_1d = close_1d - (1.1 * rng) / 2
    
    # Volume spike detection: current 4h volume > 2.0x 20-period 1d average volume
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(df_1d['volume'].values, 20)
    
    # Align 1w ADX to 4h
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Align 1d Camarilla levels to 4h (wait for 1d bar to close)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Align 1d volume MA to 4h
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_14_1w_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        is_trending = adx_14_1w_aligned[i] > 25
        
        # Volume condition: volume spike > 2.0x average
        volume_spike = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Look for reversals at extreme levels with volume spike
            # Long setup: bounce from S4 in uptring or pullback to S4 in downtrend
            if volume_spike and is_trending:
                # Long when price touches/bounces off S4 with volume
                if low[i] <= s4_1d_aligned[i] * 1.001 and close[i] > s4_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches/rejects R4 with volume
                elif high[i] >= r4_1d_aligned[i] * 0.999 and close[i] < r4_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: reversal at R4 or loss of momentum
            if high[i] >= r4_1d_aligned[i] * 0.999 or volume[i] < vol_ma_20_1d_aligned[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reversal at S4 or loss of momentum
            if low[i] <= s4_1d_aligned[i] * 1.001 or volume[i] < vol_ma_20_1d_aligned[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals