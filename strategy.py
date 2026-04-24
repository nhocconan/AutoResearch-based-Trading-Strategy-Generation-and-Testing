#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d ATR(14) for regime detection - high volatility regime (ATR > 20-period MA) for breakout follow-through.
- Volume: Current 4h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Donchian: 20-period high/low breakout levels.
- Entry: Long when price breaks above Donchian high AND 1d ATR regime bullish (ATR > ATR_MA) AND volume spike.
         Short when price breaks below Donchian low AND 1d ATR regime bullish AND volume spike.
- Exit: Price reverts to midpoint of Donchian channel or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe.
This strategy captures strong momentum moves during high volatility regimes, filtered by volume confirmation
to avoid false breakouts. Works in both bull and bear markets by taking breakouts in direction of volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime detection
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high[1:] - df_1d_low[1:]
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR(14) - exponential moving average of TR
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period ATR MA for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # 20-period volume MA for confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels on 4h data
    # Need at least 20 periods for initial calculation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # ATR regime filter: bullish when current ATR > ATR MA (volatility expanding)
    atr_regime_bullish = atr_1d_aligned > atr_ma_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian(20), ATR(14), ATR_MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike and ATR regime
            if volume_spike[i] and atr_regime_bullish[i]:
                # Bullish breakout: price > Donchian high
                if curr_close > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Donchian low
                elif curr_close < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint OR loss of volume/regime confirmation
            if (curr_close <= donchian_mid[i] or 
                not volume_spike[i] or 
                not atr_regime_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint OR loss of volume/regime confirmation
            if (curr_close >= donchian_mid[i] or 
                not volume_spike[i] or 
                not atr_regime_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0