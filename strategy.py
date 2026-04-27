#!/usr/bin/env python3
"""
4h_KAMA_Trend_12hEMA200_VolumeS
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 4h timeframe filters trend direction; combined with 12h EMA200 for long-term trend alignment and volume > 1.5x average for confirmation. 
KAMA adapts to market noise, reducing false signals in ranging markets. Works in bull markets via trend-following and in bear via avoiding false breakouts during low volatility.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag while capturing strong trends.
"""

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
    volume = prices['volume'].values
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, kama_period))  # |close[i] - close[i-kama_period]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over window
    
    # Pad arrays to match length
    change_padded = np.full(n, np.nan)
    volatility_padded = np.full(n, np.nan)
    change_padded[kama_period:] = change
    for i in range(kama_period, n):
        volatility_padded[i] = np.sum(np.abs(np.diff(close[i-kama_period+1:i+1])))
    
    er = np.where(volatility_padded > 0, change_padded / volatility_padded, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[kama_period] = close[kama_period]  # seed
    for i in range(kama_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate EMA(200) on 12h close
    close_12h = df_12h['close'].values
    ema_period = 200
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Align 12h EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need KAMA (seeded at kama_period), EMA (200), volume MA (20)
    start_idx = max(kama_period + 1, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below KAMA and 12h EMA200
        price_above_kama = price > kama[i]
        price_below_kama = price < kama[i]
        price_above_ema = price > ema_aligned[i]
        price_below_ema = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price above both KAMA and 12h EMA200 with volume
            if price_above_kama and price_above_ema and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price below both KAMA and 12h EMA200 with volume
            elif price_below_kama and price_below_ema and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA or 12h EMA200
            if price_below_kama or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above KAMA or 12h EMA200
            if price_above_kama or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_12hEMA200_VolumeS"
timeframe = "4h"
leverage = 1.0