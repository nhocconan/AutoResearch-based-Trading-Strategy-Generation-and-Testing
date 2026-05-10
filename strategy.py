#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Breakout above Camarilla R1 or below S1 with 1-week EMA50 trend filter and volume confirmation.
Works in both bull and bear markets by following weekly trend and using volatility-adjusted breakouts.
Targets 15-30 trades/year per symbol with discrete position sizing to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Calculate 1-week EMA50 for trend filter (using HTF data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 50)  # Ensure volume SMA and EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i >= 2:  # Need at least 2 bars (previous day's data on 12h chart)
            # Previous day's OHLC (assuming 2 bars per day on 12h chart)
            prev_day_start = max(0, i - 2)
            prev_day_high = np.max(high[prev_day_start:i])
            prev_day_low = np.min(low[prev_day_start:i])
            prev_day_close = close[i-1]
            
            # Camarilla levels
            range_val = prev_day_high - prev_day_low
            r1 = prev_day_close + (range_val * 1.1 / 12)
            s1 = prev_day_close - (range_val * 1.1 / 12)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R1 with uptrend and volume confirmation
            if close[i] > r1 and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with downtrend and volume confirmation
            elif close[i] < s1 and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below 1-week EMA50
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above 1-week EMA50
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals