#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d EMA50 trend filter.
# Long when Bull Power > 0 and Bear Power < 0 (both bullish) in 1d uptrend (price > 1d EMA50).
# Short when Bull Power < 0 and Bear Power > 0 (both bearish) in 1d downtrend (price < 1d EMA50).
# Uses volume confirmation (volume > 1.5x 20-period average) to filter false signals.
# Designed for 15-35 trades/year to avoid fee drag, works in both bull and bear markets by following the higher timeframe trend.

name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
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
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Both bullish (BP>0, BP<0) in 1d uptrend with volume confirmation
            if bull_power[i] > 0 and bear_power[i] > 0 and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Both bearish (BP<0, BP>0) in 1d downtrend with volume confirmation
            elif bull_power[i] < 0 and bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Either Elder Ray turns bearish or price breaks 1d EMA50
            if bull_power[i] <= 0 or bear_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Either Elder Ray turns bullish or price breaks 1d EMA50
            if bull_power[i] >= 0 or bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals