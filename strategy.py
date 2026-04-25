#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1-day EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 (close > EMA13) with 1d uptrend and volume spike.
Short when Bear Power < 0 (close < EMA13) with 1d downtrend and volume spike.
Elder Ray measures trend strength relative to short-term EMA, working in both bull and bear markets.
Volume confirmation reduces false signals. Target: 12-37 trades/year on 6h timeframe.
"""

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
    
    # Calculate Elder Ray components: Bull Power = close - EMA13, Bear Power = EMA13 - close
    # Using EMA13 for short-term trend reference
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # > 0 indicates bullish momentum
    bear_power = ema13 - close  # > 0 indicates bearish momentum
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13, EMA50(1d), and volume MA
    start_idx = max(13, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 + 1d uptrend + volume spike
            long_setup = (bull_power[i] > 0) and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: Bear Power > 0 + 1d downtrend + volume spike
            short_setup = (bear_power[i] > 0) and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR 1d trend turns down
            if (bull_power[i] <= 0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power turns negative OR 1d trend turns up
            if (bear_power[i] <= 0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0