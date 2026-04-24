#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume spike detection (>2.0x 20-period average) and choppiness regime (CHOP > 61.8 = range, < 38.2 = trend).
- Camarilla levels: R1, S1, R3, S3 calculated from prior 1d OHLC.
- Entry: Long when price > R1 AND trending regime (CHOP < 38.2) AND volume > 2.0 * 20-period average volume.
         Short when price < S1 AND trending regime AND volume confirmation.
- Exit: Opposite Camarilla breakout (price < R1 for long exit, price > S1 for short exit) OR regime shifts to choppy (CHOP > 61.8).
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
    
    # Calculate 1d choppiness index and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for CHOP
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # True Range for choppiness
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(14) for denominator
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high)-min(low)) * sqrt(period))
    # Using rolling window of 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    chop_raw = 100 * np.log10(sum_atr / range_hl * np.sqrt(14))
    chop_raw = np.where(range_hl == 0, 50, chop_raw)  # Default to middle when no range
    chop_raw = np.concatenate([[np.nan] * 13, chop_raw[13:]])  # Align for min_periods
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # R1 = C + (H-L) * 1.1/12, S1 = C - (H-L) * 1.1/12
    # R3 = C + (H-L) * 1.1/4, S3 = C - (H-L) * 1.1/4
    camarilla_multiplier = 1.1
    r1_1d = close_1d + (high_1d - low_1d) * camarilla_multiplier / 12
    s1_1d = close_1d - (high_1d - low_1d) * camarilla_multiplier / 12
    r3_1d = close_1d + (high_1d - low_1d) * camarilla_multiplier / 4
    s3_1d = close_1d - (high_1d - low_1d) * camarilla_multiplier / 4
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need 30 for CHOP calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = choppy (favor mean reversion)
        trending_regime = chop_aligned[i] < 38.2
        choppy_regime = chop_aligned[i] > 61.8
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            # Exit long: price < R1 OR regime shifts to choppy
            if position == 1:
                if curr_close < r1_aligned[i] or choppy_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S1 OR regime shifts to choppy
            elif position == -1:
                if curr_close > s1_aligned[i] or choppy_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > R1 AND trending regime AND volume confirmation
            long_condition = (curr_close > r1_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < S1 AND trending regime AND volume confirmation
            short_condition = (curr_close < s1_aligned[i] and 
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

name = "4h_Camarilla_R1S1_Breakout_1dVolSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0