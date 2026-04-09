#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume spike
# - Primary signal: 12h price breaks above/below 20-period Donchian channel
# - Regime filter: 1d ATR(14) > 20-period median ATR (only trade in volatile regimes)
# - Volume confirmation: 12h volume > 1.5x 20-period median volume
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, ATR filter avoids low-vol chop, volume confirms participation

name = "12h_1d_donchian_atr_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))),
                       np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    median_atr_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).median().values
    atr_regime = atr_14_1d > median_atr_20_1d  # Only trade when ATR > median
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Pre-compute 12h Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe (completed 12h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_regime_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR ATR regime ends
            if close[i] < donchian_low_aligned[i] or not atr_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR ATR regime ends
            if close[i] > donchian_high_aligned[i] or not atr_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ATR regime
            # Long: price breaks above Donchian high AND volume spike AND ATR regime
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                atr_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND ATR regime
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  atr_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals