#!/usr/bin/env python3
# 6h_RSI_Volume_Trend_Filter
# Hypothesis: RSI mean reversion on 6h with 1d trend filter and volume confirmation.
# Long when 1d trend up, RSI < 30 (oversold), and volume > 1.5x average.
# Short when 1d trend down, RSI > 70 (overbought), and volume > 1.5x average.
# Combines mean reversion in ranging markets with trend filter to avoid counter-trend traps.
# Volume confirms conviction. Designed for 50-150 trades over 4 years.

name = "6h_RSI_Volume_Trend_Filter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    # Calculate RSI(14) on 6h
    rsi_period = 14
    rsi = np.full_like(close, np.nan)
    if len(close) >= rsi_period + 1:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        avg_gain[rsi_period] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[0:rsi_period])
        
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi[rsi_period:] = 100 - (100 / (1 + rs[rsi_period:]))
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Align 1d indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period + 1, 50, 20)  # Need RSI, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + RSI < 30 + volume confirmation
            if trend_up and rsi[i] < 30 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + RSI > 70 + volume confirmation
            elif not trend_up and rsi[i] > 70 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or trend turns down
            if rsi[i] > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or trend turns up
            if rsi[i] < 50 or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals