#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long: Close > Donchian High(20) AND ATR(14) > ATR(50) AND volume > 1.5x 20-period avg
- Short: Close < Donchian Low(20) AND ATR(14) > ATR(50) AND volume > 1.5x 20-period avg
- Exit: Opposite Donchian breakout OR ATR(14) < ATR(50) (regime shift to low volatility)
- Uses 1d HTF for ATR calculation to avoid look-ahead and ensure completed-bar timing
- Designed for low trade frequency (15-35/year) to minimize fee drag
- Works in bull (buy breakouts above DC-H) and bear (sell breakdowns below DC-L)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR(50), 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility regime: ATR(14) > ATR(50) indicates high volatility/trending market
        vol_regime = atr_14_1d_aligned[i] > atr_50_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Close above prior Donchian High
        breakout_down = close[i] < donchian_low[i-1]  # Close below prior Donchian Low
        
        if position == 0:
            # Long: Donchian breakout up AND high volatility regime AND volume confirmation
            if breakout_up and vol_regime and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND high volatility regime AND volume confirmation
            elif breakout_down and vol_regime and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down OR low volatility regime (ATR(14) < ATR(50))
            if breakout_down or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up OR low volatility regime (ATR(14) < ATR(50))
            if breakout_up or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Regime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0