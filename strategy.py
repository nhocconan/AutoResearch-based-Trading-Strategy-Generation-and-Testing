#!/usr/bin/env python3
"""
12h_KAMA_Trend_1wTrend_VolumeSpike
Hypothesis: KAMA adapts to market noise, providing smooth trend signals. Combined with weekly trend filter and volume spike confirmation, this strategy captures strong trends while avoiding whipsaws in both bull and bear markets. The weekly timeframe provides robust trend direction, and volume spikes confirm momentum. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "12h_KAMA_Trend_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (adaptive moving average) on 12h data
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([])
    volatility = np.concatenate([np.full(er_len-1, np.nan), volatility]) if len(volatility) > 0 else np.full(len(close), np.nan)
    
    er = np.full_like(change, np.nan)
    er = np.concatenate([np.full(er_len-1, np.nan), er])
    valid = (~np.isnan(change)) & (~np.isnan(volatility[er_len-1:])) & (volatility[er_len-1:] != 0)
    er[er_len-1:] = np.where(valid, change[er_len-1:] / volatility[er_len-1:], 0)
    
    sc = np.full_like(er, np.nan)
    sc[er_len-1:] = (er[er_len-1:] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full_like(close, np.nan)
    if len(close) >= er_len:
        kama[er_len-1] = close[er_len-1]
        for i in range(er_len, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (ema_34_1w[i-1] * 33 + close_1w[i]) / 34
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(20, er_len)  # Ensure volume MA and KAMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price > KAMA AND weekly uptrend AND volume spike
            if (close[i] > kama[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price < KAMA AND weekly downtrend AND volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price < KAMA OR weekly downtrend
                if close[i] < kama[i] or close[i] < ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price > KAMA OR weekly uptrend
                if close[i] > kama[i] or close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals