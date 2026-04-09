#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Primary signal: Donchian breakout (20-period high/low) on 4h for directional bias
# - Volume filter: 1d volume > 1.5x 20-period median volume (ensures participation)
# - Regime filter: 4h chopiness index > 61.8 for mean reversion (avoid strong trends)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, chop filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_spike = volume_1d > (1.5 * median_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 4h chopiness index (to avoid strong trends)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chopiness Index: log(sum(TR)/[max(high)-min(low)]) / log(n) * 100
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = np.where((max_high - min_low) > 0, 
                    np.log10(sum_tr / (max_high - min_low)) / np.log10(14) * 100, 
                    50)
    
    # Chop regime: > 61.8 = ranging (mean reversion favorable)
    chop_regime = chop > 61.8
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop regime ends
            if close[i] < donch_low[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop regime ends
            if close[i] > donch_high[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume spike and chop regime
            # Long: price breaks above Donchian high AND volume spike AND chop regime
            if close[i] > donch_high[i] and volume_spike_aligned[i] and chop_regime[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND chop regime
            elif close[i] < donch_low[i] and volume_spike_aligned[i] and chop_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals