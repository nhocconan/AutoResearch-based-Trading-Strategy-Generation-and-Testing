#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h volume spike + ATR-based regime filter
# - Primary signal: Donchian(20) breakout on 4h timeframe - long when price > 20-period high, short when price < 20-period low
# - Volume confirmation: 12h volume > 1.5x 24-period median volume (avoid low-participation signals)
# - Regime filter: ATR(14) / ATR(50) < 0.8 (low volatility regime) to avoid whipsaws
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, volume confirmation ensures participation, low vol regime reduces false signals

name = "4h_12h_donchian_volume_regime_v1"
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
    
    # Pre-compute 12h volume regime
    volume_12h = df_12h['volume'].values
    median_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).median().values
    volume_spike = volume_12h > (1.5 * median_volume_24)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Pre-compute ATR regime (low volatility filter)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    low_vol_regime = atr_ratio < 0.8
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR volatility increases
            if close[i] < lowest_low[i] or not low_vol_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR volatility increases
            if close[i] > highest_high[i] or not low_vol_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and low volatility regime
            # Long: price breaks above Donchian high AND volume spike AND low vol regime
            if (close[i] > highest_high[i] and 
                volume_spike_aligned[i] and 
                low_vol_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND low vol regime
            elif (close[i] < lowest_low[i] and 
                  volume_spike_aligned[i] and 
                  low_vol_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals