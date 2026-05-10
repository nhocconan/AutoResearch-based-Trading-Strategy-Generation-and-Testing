#/usr/bin/env python3
"""
4h_4H_WeeklyPivot_Pullback_Momentum
Hypothesis: Buy pullbacks to weekly pivot support in uptrend, sell rallies to weekly pivot resistance in downtrend.
Uses weekly pivot levels (calculated from prior week) with 4-hour EMA trend filter and volume confirmation.
Designed to work in both bull and bear markets by following trend and fading extremes.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "4h_4H_WeeklyPivot_Pullback_Momentum"
timeframe = "4h"
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
    
    # Calculate ATR(14) for volatility measurement and stop sizing
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
    
    # Calculate 4h EMA20 for trend filter (short-term trend)
    ema_20 = np.full(n, np.nan)
    if n >= 20:
        ema_20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema_20[i] = alpha * close[i] + (1 - alpha) * ema_20[i-1]
    
    # Calculate weekly pivot points from weekly OHLC
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Weekly pivot point (P) = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Support 1 (S1) = (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    # Resistance 1 (R1) = (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    
    # Align weekly levels to 4h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Ensure EMA and volume SMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_20[i]) or \
           np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Pullback to weekly S1 in uptrend (price above EMA20) with volume
            if close[i] > weekly_s1_aligned[i] and close[i] <= weekly_pivot_aligned[i] and \
               close[i] > ema_20[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Rally to weekly R1 in downtrend (price below EMA20) with volume
            elif close[i] < weekly_r1_aligned[i] and close[i] >= weekly_pivot_aligned[i] and \
                 close[i] < ema_20[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses below weekly S1 or above weekly R1 (mean reversion)
            if close[i] < weekly_s1_aligned[i] or close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses below weekly S1 or above weekly R1 (mean reversion)
            if close[i] < weekly_s1_aligned[i] or close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals