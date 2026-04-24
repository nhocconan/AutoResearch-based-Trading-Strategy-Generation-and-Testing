#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from 12h timeframe for breakout signals.
- Breakout above upper band with volume > 1.8x 20-bar average = long signal.
- Breakdown below lower band with volume > 1.8x 20-bar average = short signal.
- Volatility filter: only trade when 12h ATR(14) is above its 50-period MA (avoid low-vol chop).
- Designed for 12h timeframe to capture multi-day swings with higher probability entries.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts in choppy markets.
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
    
    # Donchian channel (20-period) on 12h timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 20)  # Need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR > ATR_MA (avoid low-vol chop)
        vol_filter = atr[i] > atr_ma[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volatility and volume conditions are met
            if vol_filter and volume_confirm:
                # Long: price breaks above upper Donchian band
                if close[i] > high_ma[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian band
                elif close[i] < low_ma[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian band
            if close[i] < low_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper Donchian band
            if close[i] > high_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_ATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0