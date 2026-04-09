#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h volume confirmation + chop regime filter
# - Primary signal: Donchian(20) breakout on 4h timeframe - long on upper band, short on lower band
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Regime filter: 12h Choppiness Index > 61.8 for ranging markets (mean reversion at bands)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, chop filter avoids whipsaws in ranges

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume median for confirmation
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    median_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    volume_regime = align_htf_to_ltf(prices, df_12h, volume_12h > median_volume_20)
    
    # Pre-compute 12h Choppiness Index for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    atr_14 = np.zeros(len(close_12h))
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.abs(high_12h[1:] - close_12h[:-1]),
                    np.abs(low_12h[1:] - close_12h[:-1]))
    atr_14[1:] = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high_14 - lowest_low_14) != 0,
                    100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14),
                    50)
    chop_regime = align_htf_to_ltf(prices, df_12h, chop > 61.8)  # chop > 61.8 = ranging
    
    # Pre-compute 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_regime[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower OR chop regime ends (trending begins)
            if close_4h[i] < donchian_lower[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper OR chop regime ends (trending begins)
            if close_4h[i] > donchian_upper[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and chop regime
            # Long: price breaks above Donchian upper AND volume regime AND chop regime (ranging)
            if (close_4h[i] > donchian_upper[i] and 
                volume_regime[i] and 
                chop_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND volume regime AND chop regime (ranging)
            elif (close_4h[i] < donchian_lower[i] and 
                  volume_regime[i] and 
                  chop_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals