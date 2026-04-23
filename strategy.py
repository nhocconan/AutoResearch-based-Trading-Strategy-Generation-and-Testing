#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
- Uses Donchian channel breakouts for entry signals (long at upper band, short at lower band)
- 1d ATR ratio (ATR(7)/ATR(30)) defines volatility regime: only trade when ATR ratio < 1.2 (low volatility)
- Volume confirmation (> 1.3x 20-period average) ensures breakout has momentum
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in both bull and bear markets by trading breakouts in low volatility regimes
- Volume spike requirement reduces false breakouts during choppy periods
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
    
    # Calculate 1d ATR(7) and ATR(30) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(7) and ATR(30)
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr7 / atr30  # ATR ratio < 1.2 indicates low volatility regime
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian(20) channels on 4h data
    donchian_window = 20
    dc_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 30, 20)  # Donchian, ATR30, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in low volatility (ATR ratio < 1.2)
        low_vol_regime = atr_ratio_aligned[i] < 1.2
        
        # Breakout conditions
        price_above_upper = close[i] > dc_upper[i]
        price_below_lower = close[i] < dc_lower[i]
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band, low volatility, volume spike
            long_signal = (price_above_upper and 
                          low_vol_regime and
                          volume[i] > 1.3 * vol_ma[i])
            
            # Short conditions: price breaks below lower Donchian band, low volatility, volume spike
            short_signal = (price_below_lower and 
                           low_vol_regime and
                           volume[i] > 1.3 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or volatility regime change
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower band or volatility increases
                if (price_below_lower or 
                    atr_ratio_aligned[i] >= 1.2):  # Volatility regime change
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper band or volatility increases
                if (price_above_upper or 
                    atr_ratio_aligned[i] >= 1.2):  # Volatility regime change
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0