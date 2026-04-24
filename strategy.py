#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Primary timeframe: 12h to capture medium-term swings with lower fee drag.
- HTF: 1d ATR(14) for volatility filter (only trade when ATR > 20-period MA of ATR).
- Volume: Current 12h volume > 1.5 * 20-period volume MA to confirm institutional interest.
- Donchian: 20-period high/low breakout for trend following.
- Entry: Long when price > Donchian high AND volume spike AND ATR filter bullish.
         Short when price < Donchian low AND volume spike AND ATR filter bullish.
- Exit: Opposite Donchian breakout (long exits at low, short exits at high) or loss of volume/ATR filter.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy captures breakouts in volatile regimes while avoiding low-volatility whipsaws,
works in both bull and bear markets by following institutional volume and volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = df_1d_high[0] - df_1d_low[0]  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (EMA with alpha=1/14)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 20-period MA of ATR for volatility regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d volume MA for confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # ATR filter: only trade when current ATR > MA of ATR (volatile regime)
    atr_filter = atr_1d_aligned > atr_ma_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 20, 14)  # Need enough bars for Donchian, ATR MA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and ATR filter
            if volume_spike[i] and atr_filter[i]:
                # Bullish breakout: price > Donchian high
                if curr_high > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Donchian low
                elif curr_low < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price < Donchian low OR loss of volume/ATR filter
            if curr_low < donchian_low[i] or not volume_spike[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Donchian high OR loss of volume/ATR filter
            if curr_high > donchian_high[i] or not volume_spike[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeSpike_Regime_v1"
timeframe = "12h"
leverage = 1.0