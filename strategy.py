#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_volume_regime_v2
# Hypothesis: Daily strategy using weekly Camarilla pivot levels with volume confirmation and chop regime filter.
# Long: Price breaks above weekly R3 with volume > 1.5x 20-day average and CHOP(14) > 61.8 (ranging market).
# Short: Price breaks below weekly S3 with volume > 1.5x 20-day average and CHOP(14) > 61.8.
# Exit: Price returns to weekly pivot point (PP) for both long and short.
# Uses weekly Camarilla pivots as structure levels to avoid overtrading.
# Volume confirmation filters breakouts. Chop filter ensures mean-reverting environment.
# Target: 7-25 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_volume_regime_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-day)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div/0
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each weekly bar
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    
    # Resistance levels
    r3 = pp + (range_1w * 3.0 / 8.0)
    
    # Support levels
    s3 = pp - (range_1w * 3.0 / 8.0)
    
    # Align Camarilla levels to daily
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop filter: CHOP > 61.8 = ranging market (favor mean reversion)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly R3 with volume confirmation and chop filter
            if (close[i] > r3_aligned[i] and volume_confirmed and chop_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S3 with volume confirmation and chop filter
            elif (close[i] < s3_aligned[i] and volume_confirmed and chop_filter):
                position = -1
                signals[i] = -0.25
    
    return signals