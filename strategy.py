#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_1dVolSpike_Regime
Hypothesis: Trade 4h Donchian(20) breakouts with 1d EMA50 trend filter and 1d volume spike (>2.0x 20-bar MA). Uses chop regime filter (CHOP(14) > 61.8) to avoid whipsaw in ranging markets. Discrete sizing 0.25 to balance return and drawdown. Target 20-50 trades/year on 4h timeframe. Works in bull/bear via trend filter + volume confirmation + regime filter.
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
    
    # Get 1d data for HTF trend (EMA50) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 1d for HTF trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Donchian(20) on primary timeframe (4h)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index regime filter (CHOP(14))
    # CHOP = 100 * log10(sum(TR(14)) / (log14 * (HH(14) - LL(14)))) / log10(log14)
    tr1 = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr1) / (np.log10(14) * np.log10(hh14 - ll14))
    chop_regime = chop > 61.8  # choppy/ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20), EMA50(1d), volume MA(20), CHOP(14)
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high + above 1d EMA50 + 1d volume spike + NOT choppy regime
            long_setup = (close[i] > highest_20[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i] and \
                         (~chop_regime[i])
            # Short: price breaks below Donchian(20) low + below 1d EMA50 + 1d volume spike + NOT choppy regime
            short_setup = (close[i] < lowest_20[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i] and \
                          (~chop_regime[i])
            
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
            # Exit: price closes below Donchian(20) low OR below 1d EMA50
            if (close[i] < lowest_20[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Donchian(20) high OR above 1d EMA50
            if (close[i] > highest_20[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_1dVolSpike_Regime"
timeframe = "4h"
leverage = 1.0