#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla pivot levels (R1, S1) and volume spike confirmation.
- Regime: Choppiness Index (CHOP) > 61.8 = choppy (mean revert), < 38.2 = trending (breakout).
- Entry: Long when price > R1 AND trending regime (CHOP < 38.2) AND volume > 2.0 * 20-period average volume.
         Short when price < S1 AND trending regime AND volume confirmation.
- Exit: Opposite Camarilla level touch (price < R1 for long exit, price > S1 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in trending regimes, avoiding whipsaws in chop.
- Uses volume spike to confirm institutional interest in breakouts.
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
    
    # Calculate 1d Camarilla pivot levels (R1, S1) and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for CHOP calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    # R1 = Close + ((High - Low) * 1.1 / 12)
    # S1 = Close - ((High - Low) * 1.1 / 12)
    r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12.0)
    s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12.0)
    
    # Calculate Choppiness Index (CHOP) - 14 period
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(14) = smoothed TR
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop = np.zeros_like(hl_range)
    mask = (hl_range > 0) & (~np.isnan(hl_range))
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / hl_range[mask]) / np.log10(14)
    chop[~mask] = 50  # Neutral when range is zero
    
    # Align HTF indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 4h price for breakout detection
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need 30 for CHOP, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade breakouts in trending markets (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla level touch
        if position != 0:
            # Exit long: price < R1
            if position == 1:
                if curr_close < r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S1
            elif position == -1:
                if curr_close > s1_aligned[i]:
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

name = "4h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0