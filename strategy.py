#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels, volume average, and chop filter.
- Camarilla Pivot: identifies key support/resistance levels from prior 1d session.
- Entry: Long when price breaks above R1 AND volume > 1.8 * 20-period average volume AND chop > 61.8 (ranging market).
         Short when price breaks below S1 AND volume > 1.8 * 20-period average volume AND chop > 61.8.
- Exit: Opposite Camarilla breakout signal (close below R1 for longs, above S1 for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets as it captures mean reversion in ranging markets and breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Camarilla pivot: based on previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d volume average for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d chop regime filter (choppiness index)
    if len(df_1d) < 14:  # Need sufficient data for chop calculation
        return np.zeros(n)
    
    # Choppiness Index: measures whether market is choppy (ranging) or trending
    high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(
        np.maximum(df_1d['high'].values - df_1d['low'].values,
                   np.maximum(np.abs(df_1d['high'].values - df_1d['close'].values.shift(1)),
                              np.abs(df_1d['low'].values - df_1d['close'].values.shift(1))))
    ).rolling(window=14, min_periods=14).sum().values
    
    chop = 100 * np.log10(atr_sum / np.log14(high_14 - low_14)) / np.log10(14)
    # Handle edge cases where high_14 == low_14 (division by zero in log)
    chop = np.where((high_14 - low_14) > 0, chop, 50.0)  # Neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Camarilla breakout (close-based)
        if position != 0:
            # Exit long: price closes below R1
            if position == 1:
                if curr_close < r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price closes above S1
            elif position == -1:
                if curr_close > s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= r1_aligned[i] and prev_close < r1_aligned[i]
            breakout_down = curr_low <= s1_aligned[i] and prev_close > s1_aligned[i]
            
            # Volume confirmation: current volume > 1.8 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.8 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Chop regime filter: chop > 61.8 (ranging market)
            chop_regime = chop_aligned[i] > 61.8
            
            if breakout_up and volume_confirm and chop_regime:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and chop_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0