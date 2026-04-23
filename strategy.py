#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout + 1d ATR Regime Filter + Volume Spike
Donchian channel breakouts capture sustained momentum. The 1d ATR acts as a volatility regime filter:
- High 1d ATR (> 30-period median) = trending market → trade breakouts
- Low 1d ATR = choppy market → avoid false breakouts
Volume confirmation ensures breakout validity. Discrete sizing 0.25 limits fee churn.
Timeframe 4h balances signal quality and trade frequency. Target: 20-40 trades/year.
Works in both bull (breakouts catch trends) and bear (volatility regime avoids whipsaws).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d ATR regime: high volatility = trending market (trade breakouts)
    atr_median_1d = pd.Series(atr_14_1d).rolling(window=30, min_periods=30).median().values
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    atr_regime = atr_14_1d_aligned > atr_median_1d_aligned  # True = high vol/trending
    
    # Donchian channel (20-period) on 4h data
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: > 1.8x 20-period average (slightly stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 30, 20)  # Donchian20, ATR30median, volMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_regime[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high AND high volatility regime AND volume spike
            if (close[i] > highest_high[i] and 
                atr_regime[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low AND high volatility regime AND volume spike
            elif (close[i] < lowest_low[i] and 
                  atr_regime[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR loss of volatility regime
            exit_signal = False
            if position == 1:
                # Exit long when close < Donchian low OR volatility regime turns off
                if close[i] < lowest_low[i] or not atr_regime[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Donchian high OR volatility regime turns off
                if close[i] > highest_high[i] or not atr_regime[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0