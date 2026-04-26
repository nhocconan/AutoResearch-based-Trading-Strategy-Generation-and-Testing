#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend_Regime
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume spike and 12h EMA50 uptrend in choppy regime; short when breaks below low with volume spike and 12h EMA50 downtrend in choppy regime. Exit at opposite Donchian level or midpoint. Uses discrete 0.25 position size. Designed for BTC/ETH: Donchian provides structure, volume confirms breakouts, 12h EMA50 filters trend, chop regime avoids whipsaws in sideways markets. Targets 20-50 trades/year for optimal test generalization.
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    # Choppiness regime filter (14-period)
    chop = pd.Series(high).rolling(window=14, min_periods=14).max().values
    chop_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    chop_value = 100 * np.log10(sum_tr / (np.log(14) * atr_14)) / np.log(14)
    chop_value = np.where(sum_tr > 0, chop_value, 50)  # avoid div by zero
    chop_regime = chop_value > 50  # choppy when >50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian/volume, 50 for 12h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation in choppy regime
            # Long: break above Donchian high + 12h EMA50 uptrend + volume spike + chop regime
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i] and \
                       chop_regime[i]
            # Short: break below Donchian low + 12h EMA50 downtrend + volume spike + chop regime
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                        volume_spike[i] and \
                        chop_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below Donchian midpoint
            if close_val < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian midpoint
            if close_val > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend_Regime"
timeframe = "4h"
leverage = 1.0