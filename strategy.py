#!/usr/bin/env python3
# 2025-06-22 | 6h_RSI_Extremes_With_Volume_and_Trend_v2
# Hypothesis: RSI extremes (RSI<25 for long, RSI>75 for short) combined with volume spike (>2x 24-period average) and 12h EMA50 trend filter on 6h timeframe.
# Uses mean reversion in overextended markets with trend filter to avoid counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Version 2: Adjusted RSI thresholds to 25/75 and added trend filter to improve win rate in both bull and bear markets.

name = "6h_RSI_Extremes_With_Volume_and_Trend_v2"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI(14) on 6h closes
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_period + 1:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[rsi_period] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[0:rsi_period])
        
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi[rsi_period:] = 100 - (100 / (1 + rs[rsi_period:]))
    
    # Volume spike filter: current volume / 24-period average volume (24*6h = 6 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(rsi_period + 1, 24, 50)  # Ensure RSI, volume MA, and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: RSI < 25 (oversold) AND uptrend (price > EMA50) AND volume spike
            if (rsi[i] < 25 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: RSI > 75 (overbought) AND downtrend (price < EMA50) AND volume spike
            elif (rsi[i] > 75 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: RSI > 60 (overbought) OR trend reversal (price < EMA50)
            if rsi[i] > 60 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 40 (oversold) OR trend reversal (price > EMA50)
            if rsi[i] < 40 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals