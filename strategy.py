#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ATR regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 12h timeframe
# - Trend filter: 1d ATR(14) > 20-period median ATR (ensures sufficient volatility for breakout)
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture sustained moves; volume/volatility filters avoid false breakouts in low-momentum environments

name = "12h_1d_donchian_volume_atr_v1"
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
    
    # Pre-compute 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = high_1d[0] - close_1d[0]  # first bar
    tr3[0] = low_1d[0] - close_1d[0]   # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_median_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).median().values
    atr_regime = atr_14_1d > atr_median_20_1d
    
    # Align ATR regime to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Pre-compute Donchian channel on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian(20) upper and lower bands
    donch_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h volume regime: volume > 20-period median volume
    volume_12h = prices['volume'].values
    median_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_12h > median_volume_20_12h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_upper[i]) or
            np.isnan(donch_lower[i]) or
            np.isnan(atr_regime_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price closes below Donchian lower band (breakdown) OR ATR regime fails
            if close_12h[i] < donch_lower[i] or not atr_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above Donchian upper band (breakout) OR ATR regime fails
            if close_12h[i] > donch_upper[i] or not atr_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume and ATR regime confirmation
            # Long: Price closes above Donchian upper band AND volume regime AND ATR regime
            if (close_12h[i] > donch_upper[i] and 
                volume_regime[i] and 
                atr_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price closes below Donchian lower band AND volume regime AND ATR regime
            elif (close_12h[i] < donch_lower[i] and 
                  volume_regime[i] and 
                  atr_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals