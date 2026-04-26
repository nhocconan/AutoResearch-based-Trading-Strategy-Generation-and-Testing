#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirmation
Hypothesis: Elder Ray indicator (Bull Power = High - EMA13, Bear Power = Low - EMA13) combined with 1d trend filter and volume confirmation.
Long when Bull Power > 0 and rising, Bear Power < 0, in 1d uptrend with volume spike.
Short when Bear Power < 0 and falling, Bull Power > 0, in 1d downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets by following the 1d trend.
Target: 12-37 trades/year (50-150 total over 4 years) on 6h timeframe.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (using 6h close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 1.6x 20-period MA (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 13 for EMA13 + 20 for volume MA + 34 for 1d EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising (bullish momentum), Bear Power < 0, in 1d uptrend with volume spike
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling (bearish momentum), Bull Power > 0, in 1d downtrend with volume spike
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > 0 and downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR Bear Power becomes positive OR 1d trend changes to downtrend
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power turns positive OR Bull Power becomes negative OR 1d trend changes to uptrend
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0