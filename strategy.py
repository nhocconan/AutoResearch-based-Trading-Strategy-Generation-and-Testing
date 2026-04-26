#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_VolumeSpike_ChopFilter_v2
Hypothesis: 4h Camarilla pivot breakout with volume spike and choppiness regime filter.
- Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year)
- Camarilla R3/S3 levels from 1d data as significant support/resistance
- Volume confirmation: current volume > 1.5x 20-period average
- Choppiness regime: CHOP(14) between 38.2 and 61.8 to avoid whipsaws in strong trends
- Long when price breaks above R3 with volume spike in choppy market
- Short when price breaks below S3 with volume spike in choppy market
- Designed for 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading mean reversion at extreme pivot levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R3, R2, R1, PP, S1, S2, S3) from previous day
    # Camarilla formulas based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will have invalid data (rolled from last) - will be handled by min_periods later
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/2
    # R2 = Close + (High - Low) * 1.1/4
    # R1 = Close + (High - Low) * 1.1/6
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.1/6
    # S2 = Close - (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/2
    diff = prev_high - prev_low
    r3 = prev_close + diff * 1.1 / 2
    r2 = prev_close + diff * 1.1 / 4
    r1 = prev_close + diff * 1.1 / 6
    pp = (prev_high + prev_low + prev_close) / 3
    s1 = prev_close - diff * 1.1 / 6
    s2 = prev_close - diff * 1.1 / 4
    s3 = prev_close - diff * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Choppiness Index: CHOP(14) between 38.2 and 61.8 (range-bound market)
    # CHOP = 100 * log10(sum(ATR(1), 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    # Handle first bar where roll gives last value
    tr1[0] = high[0] - low[0]
    
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high - min_low
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(atr1 * 14 / chop_denominator) / np.log10(14)
    # Handle NaN values
    chop = np.where(np.isnan(chop), 50, chop)  # Default to middle range
    
    chop_range = (chop >= 38.2) & (chop <= 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 14 for CHOP)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (first day's rolled values will be NaN until we have real 1d data)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume and chop conditions
        vol_spike = volume_spike[i]
        in_chop = chop_range[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in choppy market
            if close[i] > r3_aligned[i] and vol_spike and in_chop:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in choppy market
            elif close[i] < s3_aligned[i] and vol_spike and in_chop:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below R1 (mean reversion) OR loss of volume/chop
            if close[i] < r1_aligned[i] or not (vol_spike and in_chop):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above S1 (mean reversion) OR loss of volume/chop
            if close[i] > s1_aligned[i] or not (vol_spike and in_chop):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_VolumeSpike_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0