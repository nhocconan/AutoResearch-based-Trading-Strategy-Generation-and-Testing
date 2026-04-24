#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla pivot levels, volume average, and choppiness calculation.
- Camarilla Pivots: identifies key support/resistance levels from prior day's range.
- Entry: Long when price breaks above R1 AND volume > 1.8 * 20-period average volume AND choppiness < 61.8 (trending regime).
         Short when price breaks below S1 AND volume > 1.8 * 20-period average volume AND choppiness < 61.8.
- Exit: Opposite Camarilla breakout signal (break below R1 for long, break above S1 for short).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla breaks capture institutional order flow around key levels.
- Volume confirmation ensures breakout legitimacy.
- Choppiness filter avoids ranging markets where false breakouts occur.
- Works in both bull and bear markets by trading volatility expansion from pivot levels.
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
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d choppiness index (14-period) for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    tr_sum_14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # True Range Choppiness: CHOP = 100 * log10(sum(tr14) / (max(high14)-min(low14))) / log10(14)
    max_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    min_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    choppiness = 100 * np.log10(tr_sum_14 / (max_high_14 - min_low_14)) / np.log10(14)
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for choppiness
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(choppiness_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below Camarilla S1
            if position == 1:
                if curr_low <= camarilla_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R1
            elif position == -1:
                if curr_high >= camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and choppiness filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= camarilla_r1_aligned[i] and prev_close < camarilla_r1_aligned[i-1]
            breakout_down = curr_low <= camarilla_s1_aligned[i] and prev_close > camarilla_s1_aligned[i-1]
            
            # Volume confirmation: current volume > 1.8 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.8 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Choppiness regime filter: CHOP < 61.8 (trending regime, not ranging)
            chop_regime = choppiness_aligned[i] < 61.8
            
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

name = "4h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0