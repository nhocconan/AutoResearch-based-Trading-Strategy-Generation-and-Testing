#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels (H3/L3) and volume confirmation.
- Regime: Chopiness index > 61.8 = choppy (mean reversion), < 38.2 = trending (breakout).
- Entry: Long when price > H3 AND trending regime AND volume > 1.5 * 24-period average volume.
         Short when price < L3 AND trending regime AND volume > 1.5 * 24-period average volume.
- Exit: Opposite Camarilla breakout (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in trending regimes, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    # H3 = close + (high - low) * 1.1 / 4
    # L3 = close - (high - low) * 1.1 / 4
    h3 = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    l3 = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1d volume average for confirmation (24-period)
    if len(df_1d) < 24:
        return np.zeros(n)
    
    vol_ma_24_1d = pd.Series(df_1d['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24_1d)
    
    # Calculate Chopiness Index (14-period) for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(14)
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chopiness Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # We'll use a simplified version: CHOP = 100 * log10(ATR14_sum / (HHV - LLV)) / log10(14)
    # For practical purposes, we use: CHOP > 61.8 = choppy, CHOP < 38.2 = trending
    
    # Calculate rolling sum of ATR14
    atr14_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    # Calculate HHV and LLV for 14 periods
    hhv = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llv = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = hhv - llv
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Chopiness Index
    chop = 100 * np.log10(atr14_sum / range_hl) / np.log10(14)
    
    # Align Chopiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14)  # Need 24 for volume MA, 14 for Chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_24_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: Chop < 38.2 = trending (favor breakouts), Chop > 61.8 = choppy (favor mean reversion)
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: current volume > 1.5 * 24-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_24_1d_aligned[i] if not np.isnan(vol_ma_24_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price < H3
            if position == 1:
                if curr_close < h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3
            elif position == -1:
                if curr_close > l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > H3 AND trending regime AND volume confirmation
            long_condition = (curr_close > h3_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < L3 AND trending regime AND volume confirmation
            short_condition = (curr_close < l3_aligned[i] and 
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

name = "12h_Camarilla_H3L3_Breakout_1dVolSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0