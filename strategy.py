#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volatility regime filter (ATR14 > ATR50) and volume confirmation (>1.5x 20-period average).
- Long: Close breaks above Donchian upper band (20-period high) + ATR14 > ATR50 (expanding volatility) + volume > 1.5x 20-period average
- Short: Close breaks below Donchian lower band (20-period low) + ATR14 > ATR50 (expanding volatility) + volume > 1.5x 20-period average
- Exit: Close crosses Donchian middle band (20-period median) or opposite stoploss via signal=0
- Uses price channel breakouts proven to work on SOL/ETH with volatility filter to avoid false breakouts in choppy markets
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with volatility expansion) and bear markets (volatility expansion often precedes breakdowns)
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
    
    # ATR calculation for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Donchian channels from 1d HTF data for structure
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) on daily timeframe
    donchian_h = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2  # middle band
    
    # Align Donchian levels to 12h timeframe
    donchian_h_aligned = align_htf_to_ltf(prices, df_1d, donchian_h)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1d, donchian_l)
    donchian_m_aligned = align_htf_to_ltf(prices, df_1d, donchian_m)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for volume MA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(donchian_h_aligned[i]) or
            np.isnan(donchian_l_aligned[i]) or
            np.isnan(donchian_m_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR(14) > ATR(50) (expanding volatility)
        vol_regime = atr_14[i] > atr_50[i]
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian upper band + volatility expansion + volume confirmation
            if (close[i] > donchian_h_aligned[i] and 
                vol_regime and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower band + volatility expansion + volume confirmation
            elif (close[i] < donchian_l_aligned[i] and 
                  vol_regime and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Donchian middle band (mean reversion)
            if close[i] < donchian_m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Donchian middle band (mean reversion)
            if close[i] > donchian_m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeSpike"
timeframe = "12h"
leverage = 1.0