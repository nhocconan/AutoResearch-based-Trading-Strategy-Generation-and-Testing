#!/usr/bin/env python3
# 6h_ADX_1dTrend_Volume
# Hypothesis: Uses ADX to measure trend strength on 6h timeframe, filtered by daily trend direction (HH/HL or LL/LH) and volume spike.
# Long when: daily uptrend (HH & HL), ADX > 25 (strong trend), volume > 1.5x 20-period average, and +DI > -DI.
# Short when: daily downtrend (LH & LL), ADX > 25, volume > 1.5x 20-period average, and -DI > +DI.
# Exit when ADX falls below 20 (weakening trend) or daily trend reverses.
# ADX filters out sideways markets, capturing only strong trends, which works in both bull and bear markets.
# Works in bull markets by catching strong uptrends and in bear by catching strong downtrends.

name = "6h_ADX_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1d > np.roll(high_1d, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1d > np.roll(low_1d, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1d < np.roll(high_1d, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1d < np.roll(low_1d, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous day, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- ADX calculation (trend strength) ---
    adx_period = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # Initial average
        result[period-1] = np.mean(arr[:period])
        # Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smooth(tr, adx_period)
    plus_di = 100 * smooth(plus_dm, adx_period) / atr
    minus_di = 100 * smooth(minus_dm, adx_period) / atr
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    dx[:] = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smooth(dx, adx_period)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d trend indicators to 6h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX (2*period) and volume MA(20)
    start_idx = max(2 * adx_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or
            np.isnan(minus_di_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # ADX and DI
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and adx_val > 25 and vol_spike and plus_di_val > minus_di_val:
                # Long: daily uptrend + strong trend + volume spike + bullish DI
                signals[i] = 0.25
                position = 1
            elif is_downtrend and adx_val > 25 and vol_spike and minus_di_val > plus_di_val:
                # Short: daily downtrend + strong trend + volume spike + bearish DI
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: ADX falls below 20 OR daily uptrend breaks
                if adx_val < 20 or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: ADX falls below 20 OR daily downtrend breaks
                if adx_val < 20 or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals