#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h volume confirmation + 1d chop regime filter
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets
# 12h volume spike confirms momentum for breakout continuation in trending markets
# 1d chop regime filter adapts strategy: CHOP > 61.8 = range (mean revert at extremes), CHOP < 38.2 = trending (follow momentum)
# Works in bull/bear: regime filter adapts, Williams %R captures reversals in range, volume confirms breakouts in trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_12h_1d_williamsr_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Calculate 6h Williams %R (14-period)
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_6h - lowest_low_6h) != 0,
                          -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h),
                          -50)  # neutral when range is zero
    
    # Align HTF indicators to 6h timeframe
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 12h average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_12h_aligned[i]
        
        # Regime filters
        ranging_regime = chop_1d_aligned[i] > 61.8
        trending_regime = chop_1d_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold OR regime shifts to trending
            if williams_r[i] > -20 or trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought OR regime shifts to trending
            if williams_r[i] < -80 or trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if ranging_regime:
                # Mean reversion in ranging market: enter at extremes
                if williams_r[i] <= -80:  # Oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif williams_r[i] >= -20:  # Overbought -> short
                    position = -1
                    signals[i] = -0.25
            elif trending_regime and volume_confirmed:
                # Breakout continuation in trending market with volume confirmation
                if close[i] > highest_high_6h[i]:  # Break above resistance -> long
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low_6h[i]:  # Break below support -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals