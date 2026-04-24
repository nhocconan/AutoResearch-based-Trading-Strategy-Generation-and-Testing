#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h for Donchian channels and entries/exits.
- HTF: 1d ATR for volatility regime (high volatility = breakout more reliable).
- Volume: Current 4h volume > 1.8 * 20-period 4h volume MA to avoid false breakouts.
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR > 1.2 * 50-period 1d ATR MA AND volume spike.
         Short when price breaks below Donchian(20) low AND 1d ATR > 1.2 * 50-period 1d ATR MA AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
Donchian breakouts work in both bull and bear markets by capturing strong momentum moves.
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
    
    # Calculate Donchian channels on 4h (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA on 4h
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR on 1d
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range components
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-period ATR MA for regime filter
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 4h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 4h volume MA
    volume_spike = volume > (1.8 * vol_ma_4h)
    
    # ATR regime filter: current 1d ATR > 1.2 * 50-period 1d ATR MA
    atr_regime = atr_14_aligned > (1.2 * atr_ma_50_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need enough bars for Donchian and ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and ATR regime
            if volume_spike[i] and atr_regime[i]:
                # Bullish: price breaks above Donchian high
                if curr_high > period20_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low
                elif curr_low < period20_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation
            if curr_low < period20_low[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation
            if curr_high > period20_high[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0