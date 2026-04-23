#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d Elder Ray filter and volume confirmation.
Target: 15-35 trades/year per symbol (60-140 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Williams Alligator identifies trend direction via smoothed medians (Jaw/Teeth/Lips).
Elder Ray (Bull Power/Bear Power) from 1d confirms institutional buying/selling pressure.
Volume spike filters weak breakouts. Works in bull/bear via 1d trend filter.
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
    
    # Calculate 1d Elder Ray for trend/regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # 1d trend: Bull Power > 0 and rising = bullish regime
    bull_power_ma = pd.Series(bull_power_1d).rolling(window=5, min_periods=5).mean().values
    bear_power_ma = pd.Series(bear_power_1d).rolling(window=5, min_periods=5).mean().values
    bullish_regime = (bull_power_ma > 0) & (bull_power_ma > np.roll(bull_power_ma, 1))
    bearish_regime = (bear_power_ma < 0) & (bear_power_ma < np.roll(bear_power_ma, 1))
    
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1d, bullish_regime.astype(float))
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1d, bearish_regime.astype(float))
    
    # Calculate 4h Williams Alligator (Smoothed Medians)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    median_4h = (high_4h + low_4h + close_4h) / 3
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.nanmean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_4h, 13)
    teeth = smma(median_4h, 8)
    lips = smma(median_4h, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
    alligator_long = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    alligator_short = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 4h volume > 1.8x 20-period MA (balanced to reduce trades)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: Alligator uptrend AND bullish 1d regime AND volume confirmation
            if alligator_long[i] and bullish_regime_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend AND bearish 1d regime AND volume confirmation
            elif alligator_short[i] and bearish_regime_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses or 1d regime changes
            exit_signal = False
            if position == 1:
                # Exit long on Alligator downtrend or bearish regime
                if not alligator_long[i] or bearish_regime_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on Alligator uptrend or bullish regime
                if not alligator_short[i] or bullish_regime_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dElderRay_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0