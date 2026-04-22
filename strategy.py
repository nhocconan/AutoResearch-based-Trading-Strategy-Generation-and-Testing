#!/usr/bin/env python3
"""
Hypothesis: 4h ADX-based trend strength with 12h EMA crossover and volume confirmation.
Long when ADX > 25 (trending) + EMA9 crosses above EMA21 (bullish) + volume > 1.5x average.
Short when ADX > 25 (trending) + EMA9 crosses below EMA21 (bearish) + volume > 1.5x average.
Exit when ADX < 20 (weak trend) or EMA crossover reverses.
Uses 12h EMA for trend filter to avoid whipsaw in ranging markets. Targets 20-40 trades/year.
Works in both bull (strong trends) and bear (clear downtrends) markets by filtering for trending conditions only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA9 and EMA21 for trend
    ema12h_close = df_12h['close'].values
    ema12h_9 = pd.Series(ema12h_close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema12h_21 = pd.Series(ema12h_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema12h_9_aligned = align_htf_to_ltf(prices, df_12h, ema12h_9)
    ema12h_21_aligned = align_htf_to_ltf(prices, df_12h, ema12h_21)
    
    # Calculate ADX (14-period) on 4h data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (equivalent to alpha=1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values
    atr[13] = np.mean(tr[1:14]) if n >= 14 else 0
    plus_dm_sum = np.sum(plus_dm[1:14]) if n >= 14 else 0
    minus_dm_sum = np.sum(minus_dm[1:14]) if n >= 14 else 0
    
    if n >= 14:
        atr[13] = np.mean(tr[1:14])
        plus_di[13] = 100 * plus_dm_sum / atr[13] if atr[13] != 0 else 0
        minus_di[13] = 100 * minus_dm_sum / atr[13] if atr[13] != 0 else 0
        
        # Wilder smoothing for remaining periods
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_di[i] = 100 * ((plus_di[i-1] * 13) + plus_dm[i]) / (atr[i] * 14) if atr[i] != 0 else 0
            minus_di[i] = 100 * ((minus_di[i-1] * 13) + minus_dm[i]) / (atr[i] * 14) if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(14, n):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    # ADX is smoothed DX
    if n >= 28:  # Need 14+14 periods
        adx[27] = np.mean(dx[14:28]) if n >= 28 else 0
        for i in range(28, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(28, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(ema12h_9_aligned[i]) or np.isnan(ema12h_21_aligned[i]) or
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema9_val = ema12h_9_aligned[i]
        ema21_val = ema12h_21_aligned[i]
        
        if np.isnan(ema9_val) or np.isnan(ema21_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        adx_val = adx[i]
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: ADX > 25 (trending) + EMA9 > EMA21 (bullish) + volume confirmation
            if (adx_val > 25 and ema9_val > ema21_val and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending) + EMA9 < EMA21 (bearish) + volume confirmation
            elif (adx_val > 25 and ema9_val < ema21_val and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (weak trend) or EMA9 < EMA21 (bearish crossover)
                if adx_val < 20 or ema9_val < ema21_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (weak trend) or EMA9 > EMA21 (bullish crossover)
                if adx_val < 20 or ema9_val > ema21_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_ADX_TrendStrength_12hEMA_Crossover_Volume"
timeframe = "4h"
leverage = 1.0