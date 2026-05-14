#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and choppiness regime filter.
# Uses Donchian channel (20-period high/low) from prior 1d for structure, 1w volume spike for conviction,
# and 1d choppiness index (CHOP) to avoid ranging markets. Discrete position sizing (0.0, ±0.25)
# minimizes fee churn. Designed to capture strong breakouts in trending markets while avoiding
# whipsaws in chop. Targets 15-30 trades/year per symbol.

name = "1d_Donchian20_Breakout_1wVolumeSpike_CHOPFilter_v1"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # Choppiness Index (CHOP) - range: 0-100, >61.8 = range, <38.2 = trend
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (HHV - LLV))) / log10(n)
    # Simplified: use ATR and range over 14 periods
    atr_14 = pd.Series(np.abs(high - low)).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hhvl_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    llvl_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hhvl_14 - llvl_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    chop = np.nan_to_num(chop, nan=50.0)  # fill NaN with neutral
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian levels (20-period) from prior 1w bar
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(chop[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        if chop[i] >= 61.8:
            # In choppy regime, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Trending regime: look for breakouts
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if close[i] > donchian_high_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike
            elif close[i] < donchian_low_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (mean reversion)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (mean reversion)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals