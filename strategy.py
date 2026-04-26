#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeFilter_v1
Hypothesis: 6-hour Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with daily trend filter and volume confirmation.
In bull markets (daily close > EMA50), we take longs when Bull Power > 0 and rising + volume above average.
In bear markets (daily close < EMA50), we take shorts when Bear Power > 0 and rising + volume above average.
Elder Ray measures power of bulls/bears relative to trend; volume confirms conviction.
Designed for low trade frequency (target 12-37/year) to minimize fee drag while working in both bull and bear regimes.
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
    
    # Get daily data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on daily for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA
    bear_power = ema_13 - low   # Bear Power: EMA - Low
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of daily EMA(50), 6h EMA(13), volume MA
    start_idx = max(50, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_13[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Trend filters from daily
        trend_1d_up = close_val > ema_50_1d_aligned[i]   # Daily uptrend
        trend_1d_down = close_val < ema_50_1d_aligned[i]  # Daily downtrend
        
        # Volume confirmation: above average
        volume_confirmed = vol_val > vol_ma_val
        
        if position == 0:
            # Long: daily uptrend AND Bull Power > 0 AND rising (current > previous) AND volume confirmed
            if i > start_idx:
                bull_rising = bull_val > bull_power[i-1]
                long_signal = trend_1d_up and (bull_val > 0) and bull_rising and volume_confirmed
            else:
                long_signal = False
            
            # Short: daily downtrend AND Bear Power > 0 AND rising (current > previous) AND volume confirmed
            if i > start_idx:
                bear_rising = bear_val > bear_power[i-1]
                short_signal = trend_1d_down and (bear_val > 0) and bear_rising and volume_confirmed
            else:
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: daily trend flips down OR Bull Power becomes negative
            if (not trend_1d_up) or (bull_val <= 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: daily trend flips up OR Bear Power becomes negative
            if (not trend_1d_down) or (bear_val <= 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0