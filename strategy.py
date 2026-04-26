#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_Regime
Hypothesis: On 4h timeframe, use Donchian(20) breakout confirmed by 12h EMA50 trend, volume spike, and choppy regime filter (CHOP>61.8) for mean reversion in sideways markets. 
Long when: price breaks above Donchian upper + 12h EMA50 uptrend + volume spike + choppy regime.
Short when: price breaks below Donchian lower + 12h EMA50 downtrend + volume spike + choppy regime.
Exit: reverse signal or ATR-based trailing stop (implemented via signal=0 when conditions fail).
Uses discrete 0.25 position size. Targets 20-50 trades/year on 4h for optimal generalization.
Works in bull via breakout with trend, in bear via mean reversion in choppy regimes.
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
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Chopiness index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR over period) / log10(highest high - lowest low over period)) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(hh - ll)) / np.log10(14)
    chop_regime = chop > 61.8  # choppy/sideways market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 20 for volume, 14 for chop, 50 for 12h EMA
    start_idx = max(lookback, 20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and regime confirmation
            # Long: break above Donchian high + 12h EMA50 uptrend + volume spike + choppy regime
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i] and \
                       chop_regime[i]
            # Short: break below Donchian low + 12h EMA50 downtrend + volume spike + choppy regime
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
            # Long - exit when price breaks below Donchian low (contrarian) or trend fails
            if (close_val < donchian_low[i]) or (ema_50_12h_aligned[i] <= ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high (contrarian) or trend fails
            if (close_val > donchian_high[i]) or (ema_50_12h_aligned[i] >= ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0