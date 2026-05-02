#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses Donchian channels from 4h for structure, 1d ATR(14) to distinguish trending vs ranging markets
# Volume spike ensures institutional participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 25-50 trades/year (100-200 total over 4 years) to stay within fee drag limits
# Works in bull markets via breakouts and in bear markets via breakdowns with volume confirmation
# ATR regime filter avoids whipsaws in low-volatility chop

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
    
    # Align 1d ATR to 4h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4h Donchian(20) channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 30  # max(20 for Donchian, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # ATR regime filter: only trade when volatility is elevated (avoid low-vol chop)
        # Use 10-period ATR mean to define normal volatility
        atr_ma = pd.Series(atr_14_aligned).rolling(window=10, min_periods=10).mean().shift(1).values
        if np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            volatility_filter = True  # allow trade if MA not ready
        else:
            volatility_filter = atr_14_aligned[i] > (atr_ma[i] * 1.2)  # trade when ATR > 120% of MA
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND volume confirm AND volatility filter
            if (close[i] > high_ma[i] and 
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume confirm AND volatility filter
            elif (close[i] < low_ma[i] and 
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR volatility drops to normal
            if (close[i] < low_ma[i] or 
                (not volatility_filter and atr_14_aligned[i] < (atr_ma[i] * 0.8))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR volatility drops to normal
            if (close[i] > high_ma[i] or 
                (not volatility_filter and atr_14_aligned[i] < (atr_ma[i] * 0.8))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals