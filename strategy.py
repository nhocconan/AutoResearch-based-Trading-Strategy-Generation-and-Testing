#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: Trade 6h Elder Ray Bull/Bear Power with 1d EMA trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d close > EMA(34) AND volume spike.
Short when Bull Power < 0 AND Bear Power > 0 AND 1d close < EMA(34) AND volume spike.
Uses 1d trend filter to avoid counter-trend trades in ranging markets. Targets 50-150 total trades over 4 years (12-37/year).
Works in bull (trend continuation) and bear (trend continuation) via 1d EMA filter.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA(34), 6h EMA(13), volume MA(20)
    start_idx = max(34, 13, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
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
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND volume confirm AND 1d uptrend
            long_signal = (bull_power[i] > 0) and (bear_power[i] < 0) and vol_conf and trend_up
            
            # Short: Bull Power < 0 AND Bear Power > 0 AND volume confirm AND 1d downtrend
            short_signal = (bull_power[i] < 0) and (bear_power[i] > 0) and vol_conf and trend_down
            
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
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR 1d trend flips down
            if (bull_power[i] <= 0) or (bear_power[i] >= 0) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power >= 0 OR Bear Power <= 0 OR 1d trend flips up
            if (bull_power[i] >= 0) or (bear_power[i] <= 0) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0