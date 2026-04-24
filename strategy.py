#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based volatility regime.
- ATR ratio (current ATR / 20-period ATR MA) > 1.5 indicates high volatility (breakout favorable).
- ATR ratio < 0.8 indicates low volatility (range-bound, avoid breakouts).
- Entry: Long when price breaks above Donchian(20) upper AND ATR ratio > 1.2 (breakout in expanding vol).
         Short when price breaks below Donchian(20) lower AND ATR ratio > 1.2.
         In low vol (ATR ratio < 0.8): avoid new breakout entries, only exit existing positions.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Exit: Opposite Donchian breakout or ATR ratio drops below 1.0 (vol contraction).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period ATR moving average
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    # ATR ratio: current ATR / ATR MA (avoid division by zero)
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Align 1d ATR ratio to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, lookback, 20)  # Need enough 1d bars for ATR and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals only in appropriate volatility regimes
            if volume_spike[i]:
                if atr_ratio_val > 1.2:  # High volatility: breakout strategy
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower Donchian
                    elif curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                # In low volatility (atr_ratio < 0.8), we avoid breakout entries to prevent whipsaws
                # Only trade breakouts when volatility is expanding or high
        elif position == 1:
            # Long exit: price closes below Donchian mid OR volatility contracts significantly
            if curr_close < donchian_mid[i] or atr_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR volatility contracts significantly
            if curr_close > donchian_mid[i] or atr_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0