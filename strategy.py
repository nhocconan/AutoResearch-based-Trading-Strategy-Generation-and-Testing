#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike.
- Primary timeframe: 4h for execution, HTF: 1d for EMA50 trend filter.
- Donchian levels from prior 20-period 4h: Upper = max(high[-20:]), Lower = min(low[-20:])
  Long when price breaks above Upper with volume spike, Short when price breaks below Lower with volume spike.
- Trend filter: Only trade in direction of 1d EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying Upper breakouts in uptrend, in bear via selling Lower breakouts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) levels from prior 20-period 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA50 trend
            if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
                ema50_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Long when price breaks above Donchian upper with volume spike
                    if close[i] > donchian_upper[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Short when price breaks below Donchian lower with volume spike
                    if close[i] < donchian_lower[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0