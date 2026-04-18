#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and volatility regime
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volatility regime: high volatility when ATR > 1.5x 50-day ATR mean
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    high_vol_regime = atr_14 > (1.5 * atr_ma_50)
    
    # 12h ATR(10) for stop loss and position sizing
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_10_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # 12h Donchian(20) breakout levels
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align daily volatility regime to 12h
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or 
            np.isnan(atr_10_12h[i]) or np.isnan(high_vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_regime = high_vol_regime_aligned[i] > 0.5
        
        if position == 0:
            # Enter long on Donchian breakout in high volatility regime
            if vol_regime and close[i] > donch_high_20[i]:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown in high volatility regime
            elif vol_regime and close[i] < donch_low_20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reverse signal or volatility contraction
            if not vol_regime or close[i] < donch_low_20[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse signal or volatility contraction
            if not vol_regime or close[i] > donch_high_20[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolRegime_Breakout"
timeframe = "12h"
leverage = 1.0