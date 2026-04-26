#!/usr/bin/env python3
"""
6h_ElderRay_Alligator_1dTrend_v1
Hypothesis: Combine Elder Ray (Bull/Bear Power) with Williams Alligator on 6h, filtered by 1d EMA50 trend. 
Long: Bull Power > 0, Bear Power < 0, price > Alligator Jaw, 1d uptrend (close > EMA50).
Short: Bear Power < 0, Bull Power > 0, price < Alligator Jaw, 1d downtrend (close < EMA50).
Uses ATR(20) trailing stop (2.0x) and volume confirmation (1.5x median). 
Designed for low trade frequency (<25/year) by requiring multiple confluence factors.
Works in bull markets (trend-following longs) and bear markets (trend-following shorts).
Focus on BTC/ETH as primary targets.
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA13, EMA8, EMA5 for Williams Alligator (Jaw, Teeth, Lips) on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values    # Teeth
    ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values    # Lips
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ATR(20) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: 1.5x median volume
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (50), Alligator EMAs (13), ATR (20), volume median (50)
    start_idx = max(50, 13, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(ema_8[i]) or 
            np.isnan(ema_5[i]) or
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr_20[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        jaw = ema_13[i]
        teeth = ema_8[i]
        lips = ema_5[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_20_val = atr_20[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > Jaw, 1d uptrend, volume confirmation
            long_signal = (bull_val > 0) and \
                          (bear_val < 0) and \
                          (close_val > jaw) and \
                          (close_val > ema_50_1d_val) and \
                          (volume_val > 1.5 * vol_median_val)
            # Short: Bear Power < 0, Bull Power > 0, price < Jaw, 1d downtrend, volume confirmation
            short_signal = (bear_val < 0) and \
                           (bull_val > 0) and \
                           (close_val < jaw) and \
                           (close_val < ema_50_1d_val) and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_20_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_20_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_20_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50 1d) or Alligator inversion (Teeth < Lips)
            if (low_val < long_stop) or (close_val < ema_50_1d_val) or (teeth < lips):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_20_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50 1d) or Alligator inversion (Teeth > Lips)
            if (high_val > short_stop) or (close_val > ema_50_1d_val) or (teeth > lips):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Alligator_1dTrend_v1"
timeframe = "6h"
leverage = 1.0