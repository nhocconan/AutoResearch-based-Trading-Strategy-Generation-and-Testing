#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d ATR(14) normalized by 20-period SMA to detect high/low volatility regimes.
         High volatility (ATR/SMA > 1.2) = trend-following mode (Donchian breakouts).
         Low volatility (ATR/SMA < 0.8) = mean-reversion mode (fade Donchian touches).
- Volume: Current 4h volume > 1.3 * 20-period volume MA to avoid low-volume noise.
- Entry: In high vol: Long on break above Donchian(20) upper, short on break below lower.
         In low vol: Long on touch/pullback from Donchian lower, short on touch/pullback from upper.
- Exit: Opposite Donchian level or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Combines trend-following in volatile markets with mean-reversion in calm markets,
using 1d ATR regime to adapt to changing market conditions (bull/bear/range).
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
    
    # Calculate Donchian channels (20-period) using previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = df_1d_high[0] - df_1d_low[0]  # First bar
    tr2[0] = np.abs(df_1d_high[0] - df_1d_close[0])
    tr3[0] = np.abs(df_1d_low[0] - df_1d_close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d SMA(20) of close for ATR normalization
    sma_1d_20 = pd.Series(df_1d_close).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: ATR(14) / SMA(20) - measures volatility relative to price level
    atr_ratio = atr_1d / sma_1d_20
    
    # Get 1d volume for volume confirmation (using 1d volume MA as proxy for institutional interest)
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d_20 = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    vol_ma_1d_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d_20)
    
    # Volume confirmation: current 4h volume > 1.3 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.3 * vol_ma_1d_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian(20), ATR(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Check for entry signals based on volatility regime
            if volume_spike[i]:
                # High volatility regime (trending): ATR/SMA > 1.2
                if atr_ratio_val > 1.2:
                    # Breakout entries
                    if curr_high > donchian_high[i]:
                        signals[i] = 0.25
                        position = 1
                    elif curr_low < donchian_low[i]:
                        signals[i] = -0.25
                        position = -1
                # Low volatility regime (mean-reverting): ATR/SMA < 0.8
                elif atr_ratio_val < 0.8:
                    # Mean-reversion entries: touch/pullback from bands
                    if curr_low <= donchian_low[i] * 1.001 and curr_close > donchian_low[i]:
                        signals[i] = 0.25
                        position = 1
                    elif curr_high >= donchian_high[i] * 0.999 and curr_close < donchian_high[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price touches opposite band OR loss of volume confirmation
            if curr_low <= donchian_low[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches opposite band OR loss of volume confirmation
            if curr_high >= donchian_high[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0