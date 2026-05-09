#!/usr/bin/env python3
# 4h_KAMA_Trend_Volume_Confirmation
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction,
# with volume confirmation and ATR-based stop loss. Designed to capture trends
# while avoiding whipsaws in ranging markets. Target: 20-35 trades/year per symbol.

name = "4h_KAMA_Trend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on daily close
    def kama(close, period=10, fast=2, slow=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < period:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros(n)
        er[period-1:] = change / np.where(volatility[period-1:] == 0, 1, volatility[period-1:])
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Initialize KAMA
        kama[period-1] = close[period-1]
        
        # Calculate KAMA
        for i in range(period, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_1d_prev = np.roll(kama_1d, 1)
    kama_1d_prev[0] = np.nan
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_prev_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_prev)
    
    # Calculate ATR for volatility filtering and stop loss
    def atr(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros(n)
        if n >= period:
            atr[period-1] = np.mean(tr[0:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        else:
            atr[:] = np.nan
        return atr
    
    atr_14 = atr(high, low, close, 14)
    
    # Volume ratio: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(kama_prev_aligned[i]) or \
           np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: KAMA rising (bullish) or falling (bearish)
        kama_rising = kama_aligned[i] > kama_prev_aligned[i]
        kama_falling = kama_aligned[i] < kama_prev_aligned[i]
        
        # Volume confirmation: above average volume
        volume_confirmed = volume_ratio[i] > 1.5
        
        if position == 0:
            # Enter long: KAMA rising AND volume confirmation
            if kama_rising and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling AND volume confirmation
            elif kama_falling and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA falling OR ATR-based stop loss
            if kama_falling or close[i] < (high[i] - 2.0 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA rising OR ATR-based stop loss
            if kama_rising or close[i] > (low[i] + 2.0 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals