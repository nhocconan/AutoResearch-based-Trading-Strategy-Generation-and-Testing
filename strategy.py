#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Filter_Volume
Hypothesis: KAMA (10,2) trend direction combined with RSI(14) overbought/oversold and volume confirmation.
Works in both bull and bear markets by following adaptive trend with mean-reversion entries.
Targets 12-37 trades/year to minimize fee drag on 12h timeframe.
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
    
    # Get 1-day data for trend filter and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (10,2) on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 1-period volatility
    # Pad arrays for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    vol = np.concatenate([np.full(1, np.nan), vol])
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[29] = close_1d[29]  # start at index 29 (30th value)
    for i in range(30, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Pad first element
    gain = np.concatenate([np.array([0.0]), gain])
    loss = np.concatenate([np.array([0.0]), loss])
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[14] = np.mean(gain[1:15])  # first 14 gains
    avg_loss[14] = np.mean(loss[1:15])  # first 14 losses
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from KAMA
        trend_up = close[i] > kama_aligned[i]
        trend_down = close[i] < kama_aligned[i]
        
        # RSI conditions: oversold (<30) for long, overbought (>70) for short
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Volume confirmation: >1.5x 20-period MA
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic
        long_entry = vol_confirm and trend_up and rsi_oversold
        short_entry = vol_confirm and trend_down and rsi_overbought
        
        # Exit logic: opposite RSI extreme or trend reversal
        long_exit = rsi_aligned[i] > 70 or not trend_up
        short_exit = rsi_aligned[i] < 30 or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_RSI_Filter_Volume"
timeframe = "12h"
leverage = 1.0