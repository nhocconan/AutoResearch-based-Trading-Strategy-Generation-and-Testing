#!/usr/bin/env python3
"""
4h_Donchian20_Volume_Spike_CRSI_v1
Donchian(20) breakout + CRSI(3,2,100) <10/>90 + volume spike (2x 20-bar avg) on 4h.
Long when price breaks above upper band in uptrend (price > EMA50) with volume.
Short when price breaks below lower band in downtrend (price < EMA50) with volume.
CRSI filters for extreme mean reversion conditions.
Target: 20-50 trades/year (80-200 total over 4 years).
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
    
    # === 4h EMA50 (trend filter) ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h CRSI(3,2,100) ===
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 3:
            if i == 3:
                avg_gain[i] = np.mean(gain[1:4])
                avg_loss[i] = np.mean(loss[1:4])
            else:
                avg_gain[i] = (avg_gain[i-1] * 2 + gain[i]) / 3
                avg_loss[i] = (avg_loss[i-1] * 2 + loss[i]) / 3
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI Streak(2)
    streak = np.zeros_like(close)
    streak[0] = 0
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # RSI of streak (period=2)
    streak_change = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_change > 0, streak_change, 0)
    streak_loss = np.where(streak_change < 0, -streak_change, 0)
    
    avg_streak_gain = np.zeros_like(close)
    avg_streak_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 2:
            if i == 2:
                avg_streak_gain[i] = np.mean(streak_gain[1:3])
                avg_streak_loss[i] = np.mean(streak_loss[1:3])
            else:
                avg_streak_gain[i] = (avg_streak_gain[i-1] + streak_gain[i]) / 2
                avg_streak_loss[i] = (avg_streak_loss[i-1] + streak_loss[i]) / 2
        else:
            avg_streak_gain[i] = np.nan
            avg_streak_loss[i] = np.nan
    
    rs_streak = np.where(avg_streak_loss != 0, avg_streak_gain / avg_streak_loss, 100)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Percent Rank(100) of RSI(3)
    def rolling_percentile(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                window_data = arr[i-window+1:i+1]
                valid = window_data[~np.isnan(window_data)]
                if len(valid) > 0:
                    percentile = (np.sum(valid <= arr[i]) / len(valid)) * 100
                    result[i] = percentile
        return result
    
    percent_rank = rolling_percentile(rsi3, 100)
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi3 + rsi_streak + percent_rank) / 3
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma_20 * 2.0  # volume spike: 2x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(crsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend (price > EMA50) AND CRSI < 10 AND volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema50[i] and 
                crsi[i] < 10 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian AND downtrend (price < EMA50) AND CRSI > 90 AND volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50[i] and 
                  crsi[i] > 90 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian OR CRSI > 90 (overbought)
            if (close[i] < lowest_low[i] or 
                crsi[i] > 90):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian OR CRSI < 10 (oversold)
            if (close[i] > highest_high[i] or 
                crsi[i] < 10):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_Spike_CRSI_v1"
timeframe = "4h"
leverage = 1.0