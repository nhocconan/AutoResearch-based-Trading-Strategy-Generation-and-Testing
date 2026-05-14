#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Breakout with 1d ATR Regime Filter
# Long when: Price breaks above upper BB (20,2) AND 1d ATR(14) < 1d ATR(50) (low volatility regime)
# Short when: Price breaks below lower BB (20,2) AND 1d ATR(14) < 1d ATR(50) (low volatility regime)
# Exit when price returns to middle BB (mean reversion)
# Bollinger Breakout captures volatility expansion after consolidation
# ATR regime filter ensures we only trade during low volatility periods (pre-breakout squeeze)
# Works in both bull and bear markets by trading breakouts in direction of the squeeze break
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_BollingerBreakout_ATRRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime: low volatility (ATR14 < ATR50)
        low_vol_regime = atr_14_aligned[i] < atr_50_aligned[i]
        
        if position == 0:
            # Long: Break above upper BB in low volatility regime
            if close[i] > upper_bb[i] and low_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower BB in low volatility regime
            elif close[i] < lower_bb[i] and low_vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle BB (mean reversion)
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle BB (mean reversion)
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals