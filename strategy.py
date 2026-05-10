#!/usr/bin/env python3
# 1D_VWAP_Reversion_Range_Market
# Hypothesis: In ranging markets (identified by low ADX), price reverts to the daily VWAP with high probability.
# Uses daily VWAP as mean reversion target, ADX(14) < 20 to identify ranging conditions,
# and requires price to deviate >1.5 ATR from VWAP for entry. Exits when price returns to VWAP.
# Works in both bull and bear markets by focusing on mean reversion in low volatility regimes.
# Designed for low trade frequency (~10-20/year) with discrete sizing (0.25) to minimize fee drag.

name = "1D_VWAP_Reversion_Range_Market"
timeframe = "1d"
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
    
    # Daily VWAP calculation
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    
    # Daily ATR for entry threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ADX for regime filter (trending vs ranging)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX components on weekly data
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # True Range
    w_tr1 = wh - wl
    w_tr2 = np.abs(wh - np.roll(wc, 1))
    w_tr3 = np.abs(wl - np.roll(wc, 1))
    w_tr = np.maximum(w_tr1, np.maximum(w_tr2, w_tr3))
    w_tr[0] = w_tr1[0]
    
    # Directional Movement
    w_dm_plus = np.where((wh - np.roll(wh, 1)) > (np.roll(wl, 1) - wl), np.maximum(wh - np.roll(wh, 1), 0), 0)
    w_dm_minus = np.where((np.roll(wl, 1) - wl) > (wh - np.roll(wh, 1)), np.maximum(np.roll(wl, 1) - wl, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    w_atr = wilders_smoothing(w_tr, period)
    w_dm_plus_smooth = wilders_smoothing(w_dm_plus, period)
    w_dm_minus_smooth = wilders_smoothing(w_dm_minus, period)
    
    # Directional Indicators
    w_di_plus = 100 * w_dm_plus_smooth / w_atr
    w_di_minus = 100 * w_dm_minus_smooth / w_atr
    
    # DX and ADX
    w_dx = 100 * np.abs(w_di_plus - w_di_minus) / (w_di_plus + w_di_minus)
    w_adx = wilders_smoothing(w_dx, period)
    
    # Align weekly ADX to daily timeframe
    w_adx_aligned = align_htf_to_ltf(prices, df_1w, w_adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(w_adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range condition: weekly ADX < 20 indicates ranging market
        is_ranging = w_adx_aligned[i] < 20
        
        if position == 0 and is_ranging:
            # Long entry: price below VWAP by more than 1.5*ATR
            if close[i] < vwap[i] - 1.5 * atr[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP by more than 1.5*ATR
            elif close[i] > vwap[i] + 1.5 * atr[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above VWAP
            if close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below VWAP
            if close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals