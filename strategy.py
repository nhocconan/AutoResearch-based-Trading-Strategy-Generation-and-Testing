#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day ATR regime filter.
# Long when price breaks above upper Donchian with ATR(14) > median ATR(100) (high volatility regime).
# Short when price breaks below lower Donchian with ATR(14) > median ATR(100).
# Uses volatility regime to avoid choppy markets and focus on trending periods.
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range.

name = "4h_donchian20_atr_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Median ATR(100) for regime filter
    atr_median = pd.Series(atr).rolling(window=100, min_periods=100).median().values
    
    # Volatility regime: ATR > median ATR (high volatility/trending regime)
    vol_regime = atr > atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after ATR median warmup
        # Skip if volatility regime data not available
        if np.isnan(vol_regime[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to opposite Donchian band
        if position == 1:  # long position
            if low[i] <= lower[i]:  # Exit long when price touches lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if high[i] >= upper[i]:  # Exit short when price touches upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volatility regime filter
            if vol_regime[i]:
                # Long: break above upper Donchian in high volatility regime
                if high[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian in high volatility regime
                elif low[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals