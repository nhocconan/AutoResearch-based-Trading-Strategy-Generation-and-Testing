#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) combined with weekly trend filter and volume spike.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA)
- Bullish when Bull Power > 0 AND Bear Power < 0 (market making higher highs/lows relative to trend)
- Bearish when Bull Power < 0 AND Bear Power > 0 (market making lower highs/lows)
- Weekly EMA50 trend filter ensures alignment with dominant trend (avoid counter-trend)
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Discrete position size 0.25 to manage drawdown in volatile 6h timeframe
- Target: 12-30 trades/year on 6h (50-120 total over 4 years)
- Novel combination: Elder Ray (momentum) + weekly trend + volume (proven edge in BTC/ETH)
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
    
    # Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly data for EMA50 trend filter (stronger trend filter for 6h)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 50)  # volume MA, EMA13, weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price above weekly EMA50 AND volume
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND price below weekly EMA50 AND volume
            elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Bear Power >= 0 (momentum weakening) OR price crosses below weekly EMA50
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power >= 0 OR Bear Power <= 0 (momentum weakening) OR price crosses above weekly EMA50
            if bull_power[i] >= 0 or bear_power[i] <= 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0