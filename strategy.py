#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot Breakout with 1d Volume Spike and Choppiness Regime Filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels (based on daily high/low/close) and choppiness index.
- Camarilla Pivots: Calculate R1, S1, R2, S2 from 1d OHLC.
- Entry: Long when price breaks above R1 with volume > 2.0 * 20-period average volume AND choppiness < 38.2 (trending regime).
         Short when price breaks below S1 with volume > 2.0 * 20-period average volume AND choppiness < 38.2.
- Exit: Opposite pivot break (price < S1 for long exit, price > R1 for short exit) OR choppiness > 61.8 (range regime).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla pivots work well in both trending and ranging markets when combined with regime filter.
- Volume spike confirms institutional participation.
- Choppiness index filters out sideways markets where breakouts fail.
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
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for meaningful pivots
        return np.zeros(n)
    
    # Camarilla Pivots: based on previous day's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # R2 = close + 1.1*(high-low)/6
    # S2 = close - 1.1*(high-low)/6
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    camarilla_r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Calculate 1d Choppiness Index (14-period)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Choppiness Index = 100 * log10(sum(TR14)/ (ATR14 * 14)) / log10(14)
    sum_tr14 = tr.rolling(window=14, min_periods=14).sum()
    chop = 100 * (np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14))
    chop_values = chop.values
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below S1 OR choppiness > 61.8 (range regime)
            if position == 1:
                if curr_low < s1_aligned[i] or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R1 OR choppiness > 61.8 (range regime)
            elif position == -1:
                if curr_high > r1_aligned[i] or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop filter
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Choppiness filter: only trade in trending regime (chop < 38.2)
            chop_filter = chop_aligned[i] < 38.2
            
            # Long: price breaks above R1 with volume confirmation and trending regime
            long_condition = (curr_high > r1_aligned[i] and 
                            volume_confirm and
                            chop_filter)
            
            # Short: price breaks below S1 with volume confirmation and trending regime
            short_condition = (curr_low < s1_aligned[i] and 
                             volume_confirm and
                             chop_filter)
            
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

name = "12h_Camarilla_Pivot_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0