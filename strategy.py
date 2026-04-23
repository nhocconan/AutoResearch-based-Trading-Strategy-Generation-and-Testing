#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
- Long: Price breaks above Donchian(20) high + volume > 1.3x 20-period avg + ATR(14) > ATR(50) (volatile market)
- Short: Price breaks below Donchian(20) low + volume > 1.3x 20-period avg + ATR(14) > ATR(50)
- Exit: Opposite Donchian breakout or ATR(14) < ATR(50) * 0.8 (low volatility regime)
- Uses Donchian for structure, volatility filter to avoid choppy markets, volume for conviction
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.3x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian Channel (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d ATR for additional HTF volatility context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # Need 50 for ATR50, 20 for Donchian, 14 for ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Volatility filter: ATR(14) > ATR(50) indicates volatile/trending market
        volatile_filter = atr_14[i] > atr_50[i]
        
        # Additional HTF volatility context: 1d ATR > its 20-period MA
        atr_1d_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
        htf_volatile = atr_14_1d_aligned[i] > atr_1d_ma[i] if not np.isnan(atr_1d_ma[i]) else False
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume + volatility + HTF volatile
            if (close[i] > highest_high[i] and 
                volume_confirm and 
                volatile_filter and 
                htf_volatile):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume + volatility + HTF volatile
            elif (close[i] < lowest_low[i] and 
                  volume_confirm and 
                  volatile_filter and 
                  htf_volatile):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR low volatility regime
            if (close[i] < lowest_low[i] or 
                (atr_14[i] < atr_50[i] * 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR low volatility regime
            if (close[i] > highest_high[i] or 
                (atr_14[i] < atr_50[i] * 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_ATR_Volume_1dATRFilter"
timeframe = "12h"
leverage = 1.0