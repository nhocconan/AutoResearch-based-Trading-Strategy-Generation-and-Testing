#!/usr/bin/env python3
# 1d_KAMA_Direction_1wTrend_Volume
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) direction on daily timeframe with weekly trend filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in choppy markets while tracking trends.
# Weekly trend filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation ensures trades have conviction. Designed to work in both bull and bear markets.

name = "1d_KAMA_Direction_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 20-period EMA on weekly for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === KAMA (10, 2, 30) on daily ===
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Pad volatility array to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from weekly EMA
        trend_up = close[i] > ema_20_1w_aligned[i]
        trend_down = close[i] < ema_20_1w_aligned[i]
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price above KAMA with volume and weekly uptrend
            if (price_above_kama and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA with volume and weekly downtrend
            elif (price_below_kama and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or weekly trend changes
            if (close[i] < kama[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or weekly trend changes
            if (close[i] > kama[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals