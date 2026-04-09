#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume confirmation + 1d chop regime filter
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 12h volume spike confirms breakout authenticity with higher reliability than 1h/4h volume
# 1d Choppiness Index regime filter: CHOP > 61.8 = range (mean revert at bands), CHOP < 38.2 = trending (follow breakout)
# This avoids overtrading by requiring confluence of three factors: breakout, volume, regime
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_12h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) with proper Wilder's smoothing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (1 - 1/period) * result[i-1] + (1/period) * values[i]
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
    
    # Align HTF indicators to 4h timeframe
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_volume_12h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x 12h average volume
        volume_confirmed = volume[i] > 1.8 * avg_volume_12h_aligned[i]
        
        # Regime filter: CHOP < 38.2 = strong trend (follow breakout), CHOP > 61.8 = range (mean revert)
        strong_trend = chop_1d_aligned[i] < 38.2
        ranging = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR exit ranging regime
            if close[i] < lowest_low[i] or (not strong_trend and not ranging):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR exit ranging regime
            if close[i] > highest_high[i] or (not strong_trend and not ranging):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: require volume confirmation + regime alignment
            if volume_confirmed:
                if strong_trend:
                    # Follow breakout in strong trend regime
                    if close[i] > highest_high[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -0.25
                elif ranging:
                    # Mean revert at Donchian bands in ranging regime
                    if close[i] < lowest_low[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] > highest_high[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals