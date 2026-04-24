#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter.
- Long when price breaks above Donchian(20) high AND 1d volume > 2.0 * 20-period average AND 1d chop > 61.8 (range regime)
- Short when price breaks below Donchian(20) low AND 1d volume > 2.0 * 20-period average AND 1d chop > 61.8 (range regime)
- Exit when price returns to Donchian(20) midpoint OR chop < 38.2 (trend regime)
- Uses 12h primary with 1d HTF for volume and chop filters to avoid false breakouts in low-volume/trending markets
- Designed to capture mean-reversion bounces in ranging markets (chop > 61.8) with volume confirmation
- Works in both bull and bear markets as it trades range reversals, not directional trends
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d HTF data for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for calculations
        return np.zeros(n)
    
    # 1d Volume spike filter: volume > 2.0 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(TR)/ (ATR * N)) / log10(N)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period ATR of TR)
    atr_period = 14
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of TR over N periods (for CHOP denominator)
    sum_tr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr / (atr * N)) / log10(N)
    # Where N = atr_period
    chop_denominator = atr_1d * atr_period
    # Avoid division by zero
    chop_ratio = np.where(chop_denominator > 0, sum_tr_1d / chop_denominator, 1.0)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(atr_period)
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)  # Default to neutral if undefined
    
    chop_high_regime = chop_1d > 61.8  # Ranging regime (mean reversion favorable)
    chop_low_regime = chop_1d < 38.2   # Trending regime (avoid false signals)
    chop_high_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_high_regime)
    chop_low_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_low_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 20, atr_period) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(chop_high_regime_aligned[i]) or np.isnan(chop_low_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND volume spike AND chop > 61.8 (ranging)
            if close[i] > donchian_high[i] and volume_spike_1d_aligned[i] and chop_high_regime_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND volume spike AND chop > 61.8 (ranging)
            elif close[i] < donchian_low[i] and volume_spike_1d_aligned[i] and chop_high_regime_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian mid OR chop < 38.2 (trend regime)
            if close[i] >= donchian_mid[i] or chop_low_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian mid OR chop < 38.2 (trend regime)
            if close[i] <= donchian_mid[i] or chop_low_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0