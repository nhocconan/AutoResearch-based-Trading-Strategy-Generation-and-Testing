#!/usr/bin/env python3
"""
6h_ElderRay_Reversion_1wTrend_VolumeFilter
Hypothesis: On 6h timeframe, use Elder Ray (Bull/Bear Power) from 1d for mean reversion entries, filtered by 1w trend direction (close > EMA50) and volume spike (>2.0x 20-period average). Enter long when Bear Power < 0 (bulls in control) with 1w uptrend and volume spike. Enter short when Bull Power > 0 (bears in control) with 1w downtrend and volume spike. Uses discrete position size 0.25 to balance capture and drawdown. Designed for 12-30 trades/year on 6h by requiring weekly alignment and volume confirmation, reducing overtrading while capturing reversion moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 13 or len(df_1w) < 13:  # EMA13 needs min_periods
        return np.zeros(n)
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d  # Bull Power
    bear_power = low_1d - ema_13_1d   # Bear Power
    
    # Align Elder Ray to 6h timeframe (no additional delay needed as they're based on completed 1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1w EMA warmup, volume MA warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Bear Power < 0 (bulls in control) + 1w uptrend + volume spike
            long_signal = (bear_power_aligned[i] < 0) and trend_1w_uptrend and volume_spike[i]
            
            # Short: Bull Power > 0 (bears in control) + 1w downtrend + volume spike
            short_signal = (bull_power_aligned[i] > 0) and trend_1w_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bear Power >= 0 OR 1w trend turns down
            if (bear_power_aligned[i] >= 0 or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power <= 0 OR 1w trend turns up
            if (bull_power_aligned[i] <= 0 or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Reversion_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0