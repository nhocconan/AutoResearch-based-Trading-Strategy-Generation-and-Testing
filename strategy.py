#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + choppiness regime filter
# In trending regimes (CHOP < 38.2): trade in direction of Alligator alignment (jaws < teeth < lips for long, reverse for short)
# In ranging regimes (CHOP > 61.8): fade extremes when price touches Alligator lines with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Williams Alligator is proven to work in both bull and bear markets by identifying trend strength and ranging conditions

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate SMMA for median price (typical price)
    median_price_1d = (high_1d + low_1d + close_1d) / 3.0
    jaw_raw = smma(median_price_1d, 13)
    teeth_raw = smma(median_price_1d, 8)
    lips_raw = smma(median_price_1d, 5)
    
    # Apply shifts (jaw: 8, teeth: 5, lips: 3)
    jaw_1d = np.roll(jaw_raw, 8)
    teeth_1d = np.roll(teeth_raw, 5)
    lips_1d = np.roll(lips_raw, 3)
    # Set NaN for shifted values
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Calculate 1d ATR(14) for volume normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
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
    
    # Calculate 1d average volume (20-period) normalized by ATR
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(atr_1d > 0, avg_volume_1d / atr_1d, np.nan)
    avg_vol_ratio_1d = pd.Series(vol_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 1d indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    avg_vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation array
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_confirmed = volume > 2.0 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(avg_vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Alligator loses alignment (jaws > teeth or teeth > lips)
                if jaw_1d_aligned[i] > teeth_1d_aligned[i] or teeth_1d_aligned[i] > lips_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above lips or drops below jaw
                if close[i] > lips_1d_aligned[i] or close[i] < jaw_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Alligator loses alignment (jaws < teeth or teeth < lips)
                if jaw_1d_aligned[i] < teeth_1d_aligned[i] or teeth_1d_aligned[i] < lips_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below lips or rises above jaw
                if close[i] < lips_1d_aligned[i] or close[i] > jaw_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on Alligator alignment (jaws < teeth < lips) with volume confirmation
                if jaw_1d_aligned[i] < teeth_1d_aligned[i] and teeth_1d_aligned[i] < lips_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on reverse alignment (jaws > teeth > lips) with volume confirmation
                elif jaw_1d_aligned[i] > teeth_1d_aligned[i] and teeth_1d_aligned[i] > lips_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near jaw, sell near lips when price touches Alligator lines
                if close[i] <= jaw_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= lips_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals