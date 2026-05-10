#!/usr/bin/env python3
# 6h_ElderRay_1dTrend_Volume_Spike
# Hypothesis: Combines Elder Ray (Bull/Bear power) with 1d EMA trend filter and volume confirmation.
# Long when Bull Power > 0, price above 1d EMA50, and volume > 1.5x average.
# Short when Bear Power < 0, price below 1d EMA50, and volume > 1.5x average.
# Exits when Elder Power reverses sign.
# Designed for 15-30 trades/year to avoid overtrading and work in both bull and bear markets.
# Uses 6h timeframe with 1d trend filter.

name = "6h_ElderRay_1dTrend_Volume_Spike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components (13-period EMA for power calculation)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for EMA13 and 1d EMA50
    
    for i in range(start_idx, n):
        if np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, price above 1d EMA50, volume confirmation
            if bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, price below 1d EMA50, volume confirmation
            elif bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power turns negative (trend weakness)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power turns positive (trend weakness)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals