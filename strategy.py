#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: Elder Ray Bull/Bear Power on 6h combined with 1d EMA50 trend filter and volume confirmation.
Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 1d trend alignment and volume spike.
Elder Ray measures buying/selling pressure relative to trend. In strong trends, power persists; in weak trends, power fades. Volume spike confirms conviction.
Designed for 12-30 trades/year on 6h to minimize fee drag while maintaining edge in both bull and bear markets.
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
    
    # Calculate EMA13 for Elder Ray (13-period)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying power above trend
    bear_power = low - ema13   # Selling power below trend
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for EMA13, EMA50, volume average
    start_idx = max(50, 20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Flat - look for entry: Elder Ray shows strong pressure in direction of 1d trend with volume spike
            # Long: Bull Power > 0 AND Bull Power rising (current > previous) AND 1d trend up (close > EMA50) AND volume spike
            # Short: Bear Power < 0 AND Bear Power falling (current < previous) AND 1d trend down (close < EMA50) AND volume spike
            if i > 0:
                bull_rising = bull > bull_power[i-1]
                bear_falling = bear < bear_power[i-1]
            else:
                bull_rising = False
                bear_falling = False
            
            long_condition = bull > 0 and bull_rising and close_val > ema_trend and vol_spike
            short_condition = bear < 0 and bear_falling and close_val < ema_trend and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when Bull Power turns negative (buying pressure fades) OR 1d trend turns down
            if bull <= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Bear Power turns positive (selling pressure fades) OR 1d trend turns up
            if bear >= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0