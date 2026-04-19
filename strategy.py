#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation.
# Donchian channel breakouts capture momentum moves; 1d ATR ensures volatility regime is favorable.
# Volume confirms breakout strength. Designed for 4h timeframe to capture medium-term trends
# in both bull and bear markets with low trade frequency (~20-30/year).
# Entry: Long when close > Donchian Upper(20) and ATR(1d) > median ATR(1d) and volume > 1.5x average.
# Short when close < Donchian Lower(20) and ATR(1d) > median ATR(1d) and volume > 1.5x average.
# Exit: Opposite Donchian touch or ATR drops below median (volatility collapse).
# Uses strict conditions to limit trades and avoid overtrading.

name = "4h_Donchian20_ATR1d_Volume"
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
    
    # 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range calculation for 1d
    prev_close = df_1d['close'].shift(1).values
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - prev_close)
    tr3 = np.abs(df_1d['low'].values - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Median ATR for regime filter (higher than median = volatile enough to trade)
    median_atr = np.nanmedian(atr_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volatile_regime = atr_1d_aligned > median_atr
    
    # Donchian Channel (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volatility and volume
            if (close[i] > donchian_high[i] and 
                volatile_regime[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volatility and volume
            elif (close[i] < donchian_low[i] and 
                  volatile_regime[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches Donchian low or volatility drops
            if (close[i] < donchian_low[i]) or (not volatile_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches Donchian high or volatility drops
            if (close[i] > donchian_high[i]) or (not volatile_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals