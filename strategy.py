#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses Donchian channel from price action for structure, 1d ATR(14) to identify trending vs ranging markets
# Volume spike ensures participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 15-40 trades/year (60-160 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking breakouts in trending regimes (ATR expanding)
# ATR regime filter avoids whipsaws in low volatility ranging markets

name = "4h_Donchian20_1dATR_Regime_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(50) for long-term volatility comparison
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR indicators to 4h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate 4h Donchian channel (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ATR and Donchian)
    start_idx = 50  # max(20 for Donchian, 50 for ATR50) 
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when short-term ATR > long-term ATR (expanding volatility)
        if atr_50_aligned[i] == 0:
            regime_filter = True  # allow trade if ATR50 not ready
        else:
            regime_filter = atr_14_aligned[i] > atr_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND volume confirm AND regime filter
            if (close[i] > high_ma[i] and 
                volume_confirm[i] and 
                regime_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume confirm AND regime filter
            elif (close[i] < low_ma[i] and 
                  volume_confirm[i] and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR regime changes to ranging
            if (close[i] < low_ma[i] or 
                atr_14_aligned[i] < atr_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR regime changes to ranging
            if (close[i] > high_ma[i] or 
                atr_14_aligned[i] < atr_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals