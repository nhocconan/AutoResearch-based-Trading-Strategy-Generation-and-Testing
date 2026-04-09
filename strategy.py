#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend presence and direction
# 1d volume spike (>2.0 * 20-period average volume) confirms breakout strength
# Choppiness index regime filter: CHOP < 38.2 = trending (follow Alligator), CHOP > 61.8 = range (mean revert)
# Works in bull/bear: regime filter adapts, Alligator catches strong trends, mean reversion in ranges
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator (Smoothed Medians)
    # Jaw: 13-period Smoothed Median (8-period offset)
    # Teeth: 8-period Smoothed Median (5-period offset) 
    # Lips: 5-period Smoothed Median (3-period offset)
    median_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    def smoothed_median(values, period):
        """Williams Alligator uses smoothed median (SMMA)"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_1d = smoothed_median(median_1d, 13)  # Jaw
    teeth_1d = smoothed_median(median_1d, 8)  # Teeth
    lips_1d = smoothed_median(median_1d, 5)   # Lips
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    # True Range for ATR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
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
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0 * 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        # Alligator conditions
        # Alligator asleep: jaws, teeth, lips intertwined (no trend)
        # Alligator awake: jaws, teeth, lips separated (trend)
        # Alligator eating: lips > teeth > jaws (bullish) or lips < teeth < jaws (bearish)
        bullish_alligator = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        bearish_alligator = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Alligator teeth OR regime shifts to ranging
            if close[i] < teeth_1d_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Alligator teeth OR regime shifts to ranging
            if close[i] > teeth_1d_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow Alligator in trending regime
                if bullish_alligator:
                    position = 1
                    signals[i] = 0.25
                elif bearish_alligator:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Mean revert at Alligator extremes in ranging regime
                # Buy near lips (lower extreme), sell near jaws (upper extreme)
                if close[i] < lips_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > jaw_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals