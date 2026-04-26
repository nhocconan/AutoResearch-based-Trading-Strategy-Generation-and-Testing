#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_v1
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1w trend filter.
Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) AND 1w uptrend.
Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) AND 1w downtrend.
Uses volume confirmation (1.5x average) to ensure institutional participation.
Designed for low trade frequency (12-30/year) to work in both bull (momentum continuation) and bear (mean reversion via power shifts) markets.
Discrete sizing 0.25 to manage drawdown in 2022-like crashes.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA(34), 6h EMA(13), volume MA
    start_idx = max(34, 13, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend
        trend_1w_up = close[i] > ema_34_1w_aligned[i]
        trend_1w_down = close[i] < ema_34_1w_aligned[i]
        
        # Elder Ray signals with volume confirmation
        long_signal = bull_power[i] > 0 and bear_power[i] < 0 and volume_confirm[i]
        short_signal = bull_power[i] < 0 and bear_power[i] > 0 and volume_confirm[i]
        
        if position == 0:
            if long_signal and trend_1w_up:
                signals[i] = 0.25
                position = 1
            elif short_signal and trend_1w_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 1w trend flips down OR Elder Ray turns bearish
            if not trend_1w_up or (bull_power[i] <= 0 or bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1w trend flips up OR Elder Ray turns bullish
            if not trend_1w_down or (bull_power[i] >= 0 or bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrend_v1"
timeframe = "6h"
leverage = 1.0