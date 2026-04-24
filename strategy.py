#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike filter and ATR-based regime.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume spike detection (20-period average) and ATR(10)/ATR(30) regime filter.
- Camarilla pivot levels: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2 from prior 1d candle.
- Regime: ATR(10)/ATR(30) > 1.2 = trending (favor breakouts), < 0.8 = choppy (avoid false signals).
- Entry: Long when price > R3 AND trending regime AND volume > 2.0 * 20-period average volume.
         Short when price < S3 AND trending regime AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla level (price < R3 for long exit, price > S3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in trending regimes, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(10) and ATR(30) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ATR30
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(10) and ATR(30)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio for regime: >1.2 = trending, <0.8 = choppy
    atr_ratio = atr10 / atr30
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d Camarilla levels (R3, S3) from prior day
    camarilla_window = 1
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    camarilla_range = prior_high - prior_low
    r3_level = prior_close + 1.1 * camarilla_range / 2
    s3_level = prior_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need 30 for ATR30, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade breakouts in trending markets (ATR ratio > 1.2)
        trending_regime = atr_ratio_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla level
        if position != 0:
            # Exit long: price < R3
            if position == 1:
                if curr_close < r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S3
            elif position == -1:
                if curr_close > s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > R3 AND trending regime AND volume confirmation
            long_condition = (curr_close > r3_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < S3 AND trending regime AND volume confirmation
            short_condition = (curr_close < s3_aligned[i] and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0