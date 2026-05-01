#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Uses Donchian channel breakouts from 20-period high/low for structure, filtered by 1d ATR regime
# (only trade when volatility is elevated) and volume spike for institutional participation.
# Designed to capture strong momentum moves in both bull and bear markets while avoiding choppy periods.
# Target: 25-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_Donchian20_Breakout_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar: assume prev close = open
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First bar: assume prev close = open
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian(20) channels on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 60  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volatility filter: only trade when ATR is above its 50-period median (elevated volatility)
        atr_median = np.nanmedian(atr_14_1d_aligned[max(0, i-50):i+1])
        high_volatility = atr_14_1d_aligned[i] > atr_median
        
        # Donchian breakout conditions
        breakout_up = curr_close > donchian_high[i]  # Break above upper channel
        breakout_down = curr_close < donchian_low[i]  # Break below lower channel
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper channel, volume spike, high volatility
            if breakout_up and vol_spike and high_volatility:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel, volume spike, high volatility
            elif breakout_down and vol_spike and high_volatility:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below lower channel or volatility contraction
            if curr_close < donchian_low[i] or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above upper channel or volatility contraction
            if curr_close > donchian_high[i] or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals