#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 6h timeframe, Donchian(20) breakouts in the direction of daily trend with volume spike and chop regime filter (CHOP > 61.8 = range) avoid false breakouts in choppy markets. 
Long: Price breaks above Donchian(20) high + daily uptrend + volume spike + CHOP > 61.8 (range) → mean reversion fade? Actually wait: CHOP > 61.8 = range, so we want to fade at Donchian bands? 
Better: In range (CHOP > 61.8), fade Donchian breakouts (sell at upper band, buy at lower band). In trend (CHOP < 38.2), breakout continuation.
But experiment says avoid chopped markets. So: only trade breakouts when CHOP < 61.8 (not extreme chop) OR use CHOP as trend filter: only breakout when CHOP < 50 (trending).
Let's simplify: Donchian breakout with daily trend filter + volume spike + avoid extreme chop (CHOP > 75) to prevent whipsaws in sideways markets.
Target: 12-37 trades/year.
Timeframe: 6h, HTF: 1d for trend and chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily chop regime: CHOP(14) = 100 * log10(sum(ATR(14)) / log10((HHH(14)-LLL(14)) * sqrt(14)))
    # Simplified: use ATR(14) / (highest high - lowest low over 14 days) * sqrt(14) * 100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_raw = (atr_14 / range_14) * np.sqrt(14) * 100
    chop_1d = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values  # smooth
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detector (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20)  # Donchian(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Filters
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        # Avoid extreme chop: only trade when CHOP < 75 (not too sideways)
        not_extreme_chop = chop_1d_aligned[i] < 75
        
        if position == 0:
            # Long: Price breaks above Donchian high + daily uptrend + volume spike + not extreme chop
            if close[i] > highest_high[i] and uptrend and volume_spike[i] and not_extreme_chop:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + daily downtrend + volume spike + not extreme chop
            elif close[i] < lowest_low[i] and downtrend and volume_spike[i] and not_extreme_chop:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price re-enters below Donchian high OR daily trend changes to downtrend OR extreme chop
            if close[i] < highest_high[i] or not uptrend or not not_extreme_chop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price re-enters above Donchian low OR daily trend changes to uptrend OR extreme chop
            if close[i] > lowest_low[i] or not downtrend or not not_extreme_chop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "6h"
leverage = 1.0