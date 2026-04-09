#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ATR regime filter
# - Primary signal: Donchian breakout on 12h timeframe - long when price breaks above 20-bar high, short when breaks below 20-bar low
# - Trend filter: 1d ATR ratio (ATR(7)/ATR(30)) < 1.2 to avoid high volatility chop
# - Volume confirmation: 12h volume > 1.5 * 20-period median volume (ensures participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, volume/ATR filters avoid false signals in low-participation or choppy markets

name = "12h_1d_donchian_volume_atr_v1"
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
    
    # Pre-compute 1d ATR ratio for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # ATR(7) and ATR(30)
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = np.where(atr30 != 0, atr7 / atr30, 1.0)  # avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute 12h Donchian channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h volume regime: volume > 1.5 * 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR ATR ratio > 1.5 (high volatility)
            if close_12h[i] < donchian_low[i] or atr_ratio_aligned[i] > 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR ATR ratio > 1.5 (high volatility)
            if close_12h[i] > donchian_high[i] or atr_ratio_aligned[i] > 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and ATR regime filter
            # Long: price breaks above Donchian high AND volume regime AND ATR ratio < 1.2 (low volatility)
            if (close_12h[i] > donchian_high[i] and 
                volume_regime[i] and 
                atr_ratio_aligned[i] < 1.2):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume regime AND ATR ratio < 1.2 (low volatility)
            elif (close_12h[i] < donchian_low[i] and 
                  volume_regime[i] and 
                  atr_ratio_aligned[i] < 1.2):
                position = -1
                signals[i] = -0.25
    
    return signals