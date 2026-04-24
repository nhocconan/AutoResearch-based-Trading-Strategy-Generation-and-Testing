#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike filter and ATR-based regime filter.
- Primary timeframe: 4h targeting 100-200 total trades over 4 years (25-50/year).
- HTF: 1d for Camarilla pivot levels (R3/S3) and volume confirmation.
- Regime: ATR(14)/ATR(50) ratio > 1.1 = trending (favor breakouts), < 0.9 = choppy (avoid trades).
- Entry: Long when price > R3 AND trending regime AND volume > 1.8 * 20-period average volume.
         Short when price < S3 AND trending regime AND volume > 1.8 * 20-period average volume.
- Exit: Opposite Camarilla breakout (price < R3 for long exit, price > S3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading strong breakouts in trending regimes, avoiding whipsaws in chop.
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
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR50
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
    
    # ATR(14) and ATR(50)
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio for regime: >1.1 = trending, <0.9 = choppy
    atr_ratio = atr14 / atr50
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for ATR50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade breakouts in trending markets (ATR ratio > 1.1)
        trending_regime = atr_ratio_aligned[i] > 1.1
        
        # Volume confirmation: current volume > 1.8 * 20-period average volume
        volume_confirm = curr_volume > 1.8 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price < R3
            if position == 1:
                if curr_close < camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S3
            elif position == -1:
                if curr_close > camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > R3 AND trending regime AND volume confirmation
            long_condition = (curr_close > camarilla_r3_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < S3 AND trending regime AND volume confirmation
            short_condition = (curr_close < camarilla_s3_aligned[i] and 
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