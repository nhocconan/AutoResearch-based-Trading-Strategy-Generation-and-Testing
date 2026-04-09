#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d ATR filter + volume confirmation
# - Primary signal: 4h close breaks above/below Donchian(20) channels from prior 4h bars
# - Volatility filter: Only trade when 1d ATR(14) > median ATR(14) of last 50 days (avoid low-vol chop)
# - Volume confirmation: 4h volume > 1.5x 20-period median volume (ensure participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian provides structure, ATR filter avoids ranging markets,
#   volume confirmation ensures breakout validity

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: ATR > median ATR of last 50 days
    median_atr_50 = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).median().values
    vol_regime = atr_14_1d > median_atr_50
    
    # Align volatility regime to 4h timeframe (completed 1d bar only)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels (using prior 20 bars, not including current)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 4h volume confirmation: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_regime_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR volatility regime fails
            if close[i] < lowest_20[i] or not vol_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR volatility regime fails
            if close[i] > highest_20[i] or not vol_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and volatility filter
            # Long: close breaks above Donchian high AND volume confirmation AND high volatility regime
            if close[i] > highest_20[i] and volume_confirm[i] and vol_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: close breaks below Donchian low AND volume confirmation AND high volatility regime
            elif close[i] < lowest_20[i] and volume_confirm[i] and vol_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals