#!/usr/bin/env python3
# 4h_KAMA_1dTrend_Volume
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction on 4h.
# Combined with 1d EMA34 trend filter and volume spike to avoid whipsaws.
# Long when: KAMA rising, price > KAMA, 1d EMA34 rising, volume > 1.5x 20-period avg.
# Short when: KAMA falling, price < KAMA, 1d EMA34 falling, volume > 1.5x 20-period avg.
# Exit when price crosses KAMA or 1d EMA34 trend reverses.
# Works in bull markets by catching trends and in bear by avoiding false signals via trend filter.
# KAMA reduces noise, EMA34 filters counter-trend moves, volume confirms strength.

name = "4h_KAMA_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (Adaptive Moving Average) ---
    # Parameters: ER fast=2, slow=30
    fast_sc = 2 / (2 + 1)      # 0.6667
    slow_sc = 2 / (30 + 1)     # 0.0645
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    er[10:] = change[9:] / np.maximum(volatility[9:], 1e-10)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start with first available value
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- 1d EMA34 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    # Align 1d EMA and slope to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for KAMA(10), EMA34, and volume MA(20)
    start_idx = max(10, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend conditions
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if price_above_kama and kama_rising and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: price above KAMA, KAMA rising, 1d EMA34 rising, volume spike
                signals[i] = 0.25
                position = 1
            elif price_below_kama and kama_falling and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: price below KAMA, KAMA falling, 1d EMA34 falling, volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls below KAMA OR 1d EMA34 slope turns negative
                if price_below_kama or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above KAMA OR 1d EMA34 slope turns positive
                if price_above_kama or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals