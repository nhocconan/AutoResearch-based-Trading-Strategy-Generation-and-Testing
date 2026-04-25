#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d trend filter (price > 1d EMA50 for long, < 1d EMA50 for short) and volume confirmation (>1.8x 20-bar mean volume). Long when Bull Power > 0 and rising (2-bar momentum) in uptrend with volume. Short when Bear Power > 0 and rising in downtrend with volume. Uses discrete sizing (0.25) to minimize fee churn. Targets 12-25 trades/year per symbol, effective in bull (via Bull Power strength) and bear (via Bear Power strength) markets by measuring underlying buying/selling pressure relative to trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Buying pressure
    bear_power = ema_13 - low   # Selling pressure
    
    # Calculate 2-bar momentum for power confirmation
    bull_power_mom = bull_power - np.roll(bull_power, 2)
    bear_power_mom = bear_power - np.roll(bear_power, 2)
    
    # Volume confirmation: current volume > 1.8x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising (bullish momentum) in uptrend with volume
            # Short: Bear Power > 0 and rising (bearish momentum) in downtrend with volume
            long_signal = (bull_power[i] > 0) and (bull_power_mom[i] > 0) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (bear_power[i] > 0) and (bear_power_mom[i] > 0) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Bull Power turns negative (buying pressure gone) or trend reverses
            exit_signal = (bull_power[i] <= 0) or (close[i] < ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bear Power turns negative (selling pressure gone) or trend reverses
            exit_signal = (bear_power[i] <= 0) or (close[i] > ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0