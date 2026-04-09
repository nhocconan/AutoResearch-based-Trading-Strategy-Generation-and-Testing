#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume confirmation
# - Primary signal: Donchian(20) breakout on 12h timeframe - long on upper band break, short on lower band break
# - Regime filter: 1d ATR(14) < 50-period median ATR (low volatility regime) to avoid whipsaws in choppy markets
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, ATR filter avoids false signals in ranging markets

name = "12h_1d_donchian_atr_volume_v1"
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
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    median_atr_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).median().values
    atr_regime = atr_14_1d < median_atr_50  # Low volatility regime
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Pre-compute Donchian(20) on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe (completed 12h bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_regime_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR ATR regime breaks (high volatility)
            if close[i] < donchian_lower_aligned[i] or not atr_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR ATR regime breaks (high volatility)
            if close[i] > donchian_upper_aligned[i] or not atr_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ATR regime filter
            # Long: price closes above Donchian upper band AND volume regime AND low volatility regime
            if (close[i] > donchian_upper_aligned[i] and 
                volume_regime[i] and 
                atr_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price closes below Donchian lower band AND volume regime AND low volatility regime
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_regime[i] and 
                  atr_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals