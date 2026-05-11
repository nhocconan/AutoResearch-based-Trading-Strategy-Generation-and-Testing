#!/usr/bin/env python3
# 4h_KAMA_12hTrend_Volume
# Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) on 4h to capture trend, filtered by 12h trend and volume spikes.
# Long when: 12h trend is up (KAMA rising), volume > 1.5x 20-period average, and price > KAMA (bullish).
# Short when: 12h trend is down (KAMA falling), volume > 1.5x 20-period average, and price < KAMA (bearish).
# Exit when 12h trend reverses or price crosses KAMA in opposite direction.
# Designed to work in both bull and bear markets by following the higher timeframe trend with adaptive smoothing.

name = "4h_KAMA_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h KAMA calculation ---
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=1)
    er = np.zeros_like(close_12h)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # 12h trend: rising/falling KAMA
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # --- 4h KAMA for entry trigger ---
    change_4h = np.abs(np.diff(close, n=10))
    volatility_4h = np.sum(np.abs(np.diff(close, n=1)), axis=1)
    er_4h = np.zeros_like(close)
    er_4h[10:] = change_4h[10:] / volatility_4h[10:]
    er_4h[volatility_4h == 0] = 0
    sc_4h = (er_4h * (0.6667 - 0.0645) + 0.0645) ** 2
    kama_4h = np.full_like(close, np.nan)
    kama_4h[9] = close[9]
    for i in range(10, len(close)):
        kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close[i] - kama_4h[i-1])
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 12h trend indicators to 4h timeframe
    kama_rising_aligned = align_htf_to_ltf(prices, df_12h, kama_rising)
    kama_falling_aligned = align_htf_to_ltf(prices, df_12h, kama_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for KAMA (need 10) and volume MA(20)
    start_idx = max(10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_4h[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(kama_rising_aligned[i]) or
            np.isnan(kama_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 12h
        is_kama_rising = kama_rising_aligned[i]
        is_kama_falling = kama_falling_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if is_kama_rising and vol_spike:
                # Long: 12h uptrend + volume spike + price > KAMA
                if close[i] > kama_4h[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_kama_falling and vol_spike:
                # Short: 12h downtrend + volume spike + price < KAMA
                if close[i] < kama_4h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 12h trend turns down OR price < KAMA
                if is_kama_falling or close[i] < kama_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 12h trend turns up OR price > KAMA
                if is_kama_rising or close[i] > kama_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals