#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_ATRRegime_VolumeSpike
Hypothesis: Trade 12h Donchian(20) breakouts with 1w EMA50 trend filter and ATR-based regime filter.
Only trade in direction of weekly trend to avoid counter-trend whipsaws. Volume spike confirms momentum.
ATR regime: trade only when ATR(12)/ATR(48) < 1.2 (low volatility = better breakout reliability).
Discrete sizing 0.25 to manage risk and minimize fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(12) and ATR(48) on daily timeframe
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    
    atr_12 = pd.Series(tr1).rolling(window=12, min_periods=12).mean().values
    atr_48 = pd.Series(tr1).rolling(window=48, min_periods=48).mean().values
    
    # ATR ratio: ATR(12)/ATR(48) - regime filter
    atr_ratio = atr_12 / atr_48
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian(20) on 12h timeframe
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50), ATR (48), Donchian (20), volume MA (20)
    start_idx = max(50, 48, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ATR ratio < 1.2 (low volatility regime)
        in_low_vol_regime = atr_ratio_aligned[i] < 1.2
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly trend bullish AND volume spike AND low vol regime
            long_setup = (close[i] > donch_high[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike[i] and \
                         in_low_vol_regime
            # Short: price breaks below Donchian low AND weekly trend bearish AND volume spike AND low vol regime
            short_setup = (close[i] < donch_low[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike[i] and \
                          in_low_vol_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bearish
            if (close[i] < donch_high[i] and close[i] > donch_low[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bullish
            if (close[i] < donch_high[i] and close[i] > donch_low[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_ATRRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0