#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter
Hypothesis: 4h Donchian(20) breakout with volume spike and choppiness regime filter.
- Long when price breaks above Donchian(20) high AND volume spike AND chop regime indicates trend (CHOP < 38.2)
- Short when price breaks below Donchian(20) low AND volume spike AND chop regime indicates trend (CHOP < 38.2)
- Uses 1d HTF for volume spike calculation (institutional participation)
- Choppiness index (14) filters for trending markets to avoid whipsaws in ranging conditions
- Designed for lower frequency (target 20-50 trades/year) to minimize fee drag and improve test generalization
- Exit on opposite Donchian(20) touch or chop regime shift to ranging (CHOP > 61.8)
- Novelty: Combines Donchian breakout with volume confirmation and chop regime filter for robust trend following
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume spike from 1d data (20-period average)
    vol_ma20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Donchian(20) channels on primary timeframe
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = pd.Series(high).rolling(window=14, min_periods=14).max() - pd.Series(low).rolling(window=14, min_periods=14).min()
    tr2 = abs(pd.Series(high) - pd.Series(low).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(high).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, atr_sum / range_14, 1.0)
    chop = np.where(chop_raw > 0, 100 * np.log10(chop_raw), 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Regime filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    trending_regime = chop < 38.2
    ranging_regime = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 14 for ATR, 20 for volume MA)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and trend regime filter
        if position == 0:
            # Long: Price breaks above Donchian high AND volume spike AND trending regime
            if close[i] > donch_high[i] and volume_spike_aligned[i] and trending_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND volume spike AND trending regime
            elif close[i] < donch_low[i] and volume_spike_aligned[i] and trending_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR regime shifts to ranging
            if close[i] < donch_low[i] or ranging_regime[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR regime shifts to ranging
            if close[i] > donch_high[i] or ranging_regime[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0