#!/usr/bin/env python3
"""
12h_1w_hma_adx_volume_v1
Hypothesis: 12-hour strategy using weekly HMA21 for trend direction and ADX for trend strength, with volume confirmation on breakouts.
Works in bull/bear by requiring strong trend (ADX > 25) and alignment with weekly HMA21, avoiding choppy markets.
Targets 15-30 trades per year (60-120 total over 4 years) to minimize fee drag.
"""

name = "12h_1w_hma_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half, adjust=False, min_periods=half).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean()
    raw = 2 * wma2 - wma1
    hma = pd.Series(raw).ewm(span=sqrt, adjust=False, min_periods=sqrt).mean()
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly HMA21 for trend direction
    hma21_1w = calculate_hma(close_1w, 21)
    hma21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma21_1w)
    
    # Weekly ADX for trend strength
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: volume > 2.0x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(hma21_1w_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        # Long entry: price above HMA21 (uptrend) + strong trend + volume
        if (close[i] > hma21_1w_aligned[i] and strong_trend and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below HMA21 (downtrend) + strong trend + volume
        elif (close[i] < hma21_1w_aligned[i] and strong_trend and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend weakness or price crosses back to opposite side of HMA
        elif position == 1 and (close[i] < hma21_1w_aligned[i] or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > hma21_1w_aligned[i] or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals