#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume spike filter and EMA trend.
# Uses Donchian channel (20-period) for breakout structure, ATR-normalized volume spike (>1.5x 20-bar ATR-scaled avg volume) for conviction,
# and EMA(50) on 1d to filter trend direction. Only long in 1d uptrend (price > EMA50), only short in 1d downtrend (price < EMA50).
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Designed to capture strong breakouts with volume confirmation in trending markets.
# Targets 20-50 trades/year per symbol to avoid fee drag while maintaining edge in both bull and bear regimes.

name = "4h_Donchian20_Breakout_1dATRVolumeSpike_EMATrend_v2"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
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
    volume_spike = vol_atr_ratio > (1.5 * vol_atr_ma_20)
    
    # Donchian Channel (20)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(volume_spike[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only long in 1d uptrend, only short in 1d downtrend
        if close[i] > ema_50_1d_aligned[i]:  # 1d uptrend
            # LONG: Price breaks above Donchian upper AND volume spike
            if position == 0 and close[i] > highest_20[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # EXIT LONG: Price crosses below Donchian lower
            elif position == 1 and close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            # HOLD LONG
            elif position == 1:
                signals[i] = 0.25
            # FLAT or REVERSE
            else:
                signals[i] = 0.0
        else:  # 1d downtrend (close[i] <= ema_50_1d_aligned[i])
            # SHORT: Price breaks below Donchian lower AND volume spike
            if position == 0 and close[i] < lowest_20[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            # EXIT SHORT: Price crosses above Donchian upper
            elif position == -1 and close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            # HOLD SHORT
            elif position == -1:
                signals[i] = -0.25
            # FLAT or REVERSE
            else:
                signals[i] = 0.0
    
    return signals