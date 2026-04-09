#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + volume spike + ATR-based regime filter
# - Primary signal: 4h close breaks above/below Donchian(20) channel from prior 20 bars
# - Regime filter: ATR(14)/ATR(50) > 1.2 indicates elevated volatility (breakout-prone)
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian channels adapt to volatility, volume/ATR filters ensure
#   participation and regime suitability, reducing false breakouts in low-volatility/choppy markets

name = "4h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) - using prior 20 bars only
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # ATR(14) and ATR(50) for regime filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(np.roll(close, 1) - low)
    tr = np.maximum(tr1, tr2)
    tr[0] = 0  # First bar has no prior close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.where(atr_50 == 0, np.nan, atr_50)  # Avoid division by zero
    
    # Volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR ATR ratio drops below 0.8 (low vol)
            if close[i] < lowest_low[i] or atr_ratio[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR ATR ratio drops below 0.8 (low vol)
            if close[i] > highest_high[i] or atr_ratio[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ATR regime filter
            # Long: close breaks above highest high AND volume regime AND elevated volatility
            if close[i] > highest_high[i] and volume_regime[i] and atr_ratio[i] > 1.2:
                position = 1
                signals[i] = 0.25
            # Short: close breaks below lowest low AND volume regime AND elevated volatility
            elif close[i] < lowest_low[i] and volume_regime[i] and atr_ratio[i] > 1.2:
                position = -1
                signals[i] = -0.25
    
    return signals