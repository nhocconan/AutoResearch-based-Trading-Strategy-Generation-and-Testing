#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Donchian breakouts capture institutional momentum. ATR regime filter (ATR(14)/ATR(50)) avoids
# choppy markets where breakouts fail. Volume confirmation ensures breakout validity.
# Works in bull (strong breakouts) and bear (breakdowns with volume). Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATR_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR regime: trending when ATR(14) > ATR(50) * 1.2
    atr_ratio = atr_14 / atr_50
    atr_regime = atr_ratio > 1.2
    
    # Align ATR regime to 4h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for ATR(50) + 20 for Donchian + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions (avoid look-ahead by using prior bar levels)
        breakout_up = curr_close > donchian_high[i-1]   # Break above upper channel
        breakout_down = curr_close < donchian_low[i-1]  # Break below lower channel
        
        vol_spike = volume_spike[i]
        regime = atr_regime_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, trending regime
            if breakout_up and vol_spike and regime:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, trending regime
            elif breakout_down and vol_spike and regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or regime change to choppy
            if curr_close < donchian_low[i] or not regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or regime change to choppy
            if curr_close > donchian_high[i] or not regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals