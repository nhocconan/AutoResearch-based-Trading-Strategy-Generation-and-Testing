#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + chop regime filter
# Williams Alligator (Jaw/Teeth/Lips SMAs) identifies trend direction and strength
# 1d volume spike confirms institutional participation (avoids false signals)
# Choppiness index regime filter: CHOP > 61.8 = range (fade extremes), CHOP < 38.2 = trending (follow Alligator)
# Works in bull/bear: regime filter adapts, Alligator catches strong moves while avoiding whipsaws in ranges
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
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - smoothed TR using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
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
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smoothed_mma(values, period):
        """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
        return wilders_smoothing(values, period)
    
    jaw_raw = smoothed_mma(close, 13)
    teeth_raw = smoothed_mma(close, 8)
    lips_raw = smoothed_mma(close, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # First values become NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 1d average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_1d_aligned[i]
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Alligator lines reverse (Lips < Teeth < Jaw) OR regime shifts to ranging
            if lips[i] < teeth[i] < jaw[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines reverse (Lips > Teeth > Jaw) OR regime shifts to ranging
            if lips[i] > teeth[i] > jaw[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow Alligator alignment in trending regime
                # Long: Lips > Teeth > Jaw (bullish alignment)
                # Short: Lips < Teeth < Jaw (bearish alignment)
                if lips[i] > teeth[i] > jaw[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif lips[i] < teeth[i] < jaw[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at extremes in ranging regime
                # Long when price touches lower extreme (Lips lowest)
                # Short when price touches upper extreme (Lips highest)
                if lips[i] == np.minimum(lips[i], np.minimum(teeth[i], jaw[i])) and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif lips[i] == np.maximum(lips[i], np.maximum(teeth[i], jaw[i])) and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals