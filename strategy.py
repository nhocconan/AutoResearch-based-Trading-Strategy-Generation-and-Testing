#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ATR-based trend filter
# Donchian(20) captures price structure and breakouts with clear levels.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# ATR(14) trend filter (price vs ATR-modified mean) adapts to volatility and avoids chop.
# Designed for fewer, higher-quality trades (target: 20-40/year) to minimize fee drag.
# Works in bull/bear via breakout symmetry and volatility-adjusted filters.

name = "4h_Donchian20_Breakout_VolumeSpike_ATRTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    # ATR(14) for trend filter and volatility normalization
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - np.roll(close, 1)))
    tr3 = pd.Series(abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # ATR-modified mean: close ± 0.5*ATR for dynamic trend bias
    atr_mean = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    long_filter = close > (atr_mean + 0.5 * atr)
    short_filter = close < (atr_mean - 0.5 * atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20)  # 20 for Donchian and EWM
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_mean[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        long_bias = long_filter[i]
        short_bias = short_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high, volume spike, long bias
            if close[i] > highest_high[i] and vol_spike and long_bias:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, volume spike, short bias
            elif close[i] < lowest_low[i] and vol_spike and short_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below Donchian low or loss of long bias
            if close[i] < lowest_low[i] or not long_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above Donchian high or loss of short bias
            if close[i] > highest_high[i] or not short_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals