#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Primary signal: 12h price breaks above/below 20-period Donchian channel
# - Trend filter: 12h price must be above/below 50-period EMA for confirmation
# - Volume confirmation: 1d volume > 1.5x 20-period median volume (institutional participation)
# - Regime filter: 12h choppiness index < 38.2 (trending market) OR > 61.8 (ranging market) for mean reversion at extremes
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian captures trends, chop regime adapts to market conditions, volume confirms validity

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
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
    vol_1d = df_1d['volume'].values
    median_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    volume_spike = vol_1d > (1.5 * median_vol_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h Choppiness Index (14-period)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(hh - ll)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh - ll) > 0, chop, 50.0)  # Default to neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50[i]) or np.isnan(chop[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian mid OR below EMA50
            if close[i] < donchian_mid[i] or close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian mid OR above EMA50
            if close[i] > donchian_mid[i] or close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume spike and chop regime filter
            # Long: price breaks above Donchian high AND volume spike AND (trending OR extreme ranging)
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] and 
                (chop[i] < 38.2 or chop[i] > 61.8)):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND (trending OR extreme ranging)
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] and 
                  (chop[i] < 38.2 or chop[i] > 61.8)):
                position = -1
                signals[i] = -0.25
    
    return signals