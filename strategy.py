#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + chop regime filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# 1d volume spike confirms institutional participation
# Chop regime filter adapts to market conditions (trending vs ranging)
# Designed for 12h timeframe to capture medium-term swings with low trade frequency
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_williams_alligator_volume_chop_v1"
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
    
    # Wilder's smoothing for ATR
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
    # Jaw: 13-period SMMA, smoothed by 8 periods
    # Teeth: 8-period SMMA, smoothed by 5 periods  
    # Lips: 5-period SMMA, smoothed by 3 periods
    def smoothed_mma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period - 1) + values[i]) / period
        return result
    
    # Jaw (13,8)
    jaw_raw = smoothed_mma(close, 13)
    jaw = smoothed_mma(jaw_raw, 8)
    
    # Teeth (8,5)
    teeth_raw = smoothed_mma(close, 8)
    teeth = smoothed_mma(teeth_raw, 5)
    
    # Lips (5,3)
    lips_raw = smoothed_mma(close, 5)
    lips = smoothed_mma(lips_raw, 3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow Alligator alignment), CHOP > 61.8 = range (fade extremes)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Alligator lines cross in bearish order OR regime shifts to ranging
            if (jaw[i] > teeth[i] > lips[i]) or ranging_regime:  # Bearish alignment
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross in bullish order OR regime shifts to ranging
            if (jaw[i] < teeth[i] < lips[i]) or ranging_regime:  # Bullish alignment
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow Alligator alignment in trending regime
                if (jaw[i] < teeth[i] < lips[i]) and volume_confirmed:  # Bullish alignment
                    position = 1
                    signals[i] = 0.25
                elif (jaw[i] > teeth[i] > lips[i]) and volume_confirmed:  # Bearish alignment
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Fade extremes in ranging regime: buy near Lips, sell near Jaw
                if close[i] < lips[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > jaw[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals