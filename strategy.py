#!/usr/bin/env python3
# 1d_RSI_RangeReversal_1wTrend_Filter
# Hypothesis: On daily timeframe, take mean-reversion trades at RSI extremes (RSI<30 for long, RSI>70 for short)
# only when aligned with weekly trend (price above/below weekly EMA20). Uses volume confirmation (>1.5x 20-day average)
# to filter weak signals. Designed for low trade frequency (<25/year) to minimize fee drag.
# Works in bull markets by taking pullbacks in uptrends and in bear markets by selling rallies in downtrends.

name = "1d_RSI_RangeReversal_1wTrend_Filter"
timeframe = "1d"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder's smoothing
    for i in range(len(close)):
        if i < 14:
            continue
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.full_like(close, np.nan)
    valid = avg_loss != 0
    rsi[valid] = 100 - (100 / (1 + avg_gain[valid] / avg_loss[valid]))
    
    # Volume confirmation: current volume / 20-day average volume
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
    
    start_idx = max(30, 20)  # Ensure RSI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold (<30) AND uptrend (price > weekly EMA20) AND volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) AND downtrend (price < weekly EMA20) AND volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>=50) OR trend reversal (price < weekly EMA20)
            if rsi[i] >= 50 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<=50) OR trend reversal (price > weekly EMA20)
            if rsi[i] <= 50 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals