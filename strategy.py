#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from 4h timeframe as structure.
- Breakout above upper band with volume > 2.0x 20-bar average = long signal.
- Breakdown below lower band with volume > 2.0x 20-bar average = short signal.
- Volatility filter: only trade when 1d ATR(14) is above its 50-period MA (high volatility regime).
- Designed for 4h timeframe to capture medium-term swings with proven edge.
- Uses discrete position size 0.30 to balance return and drawdown.
- Targets 20-50 trades/year (80-200 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts; volatility filter avoids choppy low-vol periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period MA for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR and its MA to 4h timeframe (wait for 1d bar to close)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Volatility filter: trade only when current ATR > its MA (high vol regime)
    vol_regime = atr_14_1d_aligned > atr_ma_50_1d_aligned
    
    # 4h Donchian(20) - calculate directly on 4h prices
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need enough for ATR MA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) and volatility regime
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        high_volatility = vol_regime[i]
        
        if position == 0:
            # Only trade if volume confirms AND high volatility regime
            if volume_confirm and high_volatility:
                # Long: price breaks above Donchian upper band
                if close[i] > donchian_high[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below Donchian lower band
                elif close[i] < donchian_low[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian lower band
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price crosses above Donchian upper band
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0