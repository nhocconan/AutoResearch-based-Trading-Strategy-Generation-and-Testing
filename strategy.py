#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volume spike and EMA trend filter.
# Uses Donchian channel (20-period high/low) from prior 12h for breakout structure,
# ATR-normalized volume spike (>1.8x 20-bar ATR-scaled avg volume) for conviction,
# and EMA(50) > EMA(200) on 1d for bullish trend bias (long only) or EMA(50) < EMA(200) for bearish bias (short only).
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Designed to capture strong breakouts
# in trending markets while avoiding false signals in ranging conditions. Targets 15-30 trades/year per symbol.

name = "12h_Donchian20_Breakout_1dATRVolumeSpike_EMATrend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # ATR(14) for volatility normalization
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (1.8 * vol_atr_ma_20)
    
    # Donchian channel (20) on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(50) and EMA(200) on 1d for trend bias
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 12h (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(atr_14[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA crossover
        bullish_bias = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        bearish_bias = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Look for breakout entries with volume confirmation and trend bias
            if bullish_bias and close[i] > highest_20[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_bias and close[i] < lowest_20[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals