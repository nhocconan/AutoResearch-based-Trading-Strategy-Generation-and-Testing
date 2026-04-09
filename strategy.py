#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 4h
# - Volume confirmation: 4h volume > 20-period median volume (avoid low-participation breakouts)
# - Regime filter: 1d choppiness index > 61.8 (range market) for mean reversion at channel edges
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, chop filter avoids false signals in strong trends

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for chop regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 1d Choppy Index: 100 * log10(sum(TR,14) / (14 * (HHV - LLV))) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = 14 * (hh_14 - ll_14)
    chop_raw = np.where(
        denominator > 0,
        100 * np.log10(atr_14 / denominator) / np.log10(14),
        50.0  # neutral when no range
    )
    
    # Align 1d chop to 4h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian middle OR chop regime ends (trending market)
            middle = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if close[i] < middle or chop_aligned[i] < 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian middle OR chop regime ends (trending market)
            middle = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if close[i] > middle or chop_aligned[i] < 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and chop regime
            # Long: price breaks above upper band AND volume regime AND chop > 61.8 (range)
            if close[i] > highest_high_20[i] and volume_regime[i] and chop_aligned[i] > 61.8:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band AND volume regime AND chop > 61.8 (range)
            elif close[i] < lowest_low_20[i] and volume_regime[i] and chop_aligned[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals