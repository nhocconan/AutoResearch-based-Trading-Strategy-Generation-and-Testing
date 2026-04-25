#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d ATR Filter and Volume Spike
Hypothesis: Donchian(20) breakouts on 12h charts capture strong momentum moves.
Price must close outside 20-period Donchian channel with volume confirmation and
1d ATR-based volatility filter to avoid choppy markets. Discrete sizing (0.25)
targets ~50-150 trades over 4 years to minimize fee drag. Works in bull/bear via
volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20)
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_value = atr_1d_aligned[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # Volatility filter: avoid extremely low volatility (chop) and extreme volatility
        # Use 1d ATR relative to its 50-period average
        if i >= 50:
            atr_ma_50 = np.mean(atr_1d_aligned[max(0, i-49):i+1])
            vol_regime = atr_value > 0.5 * atr_ma_50 and atr_value < 3.0 * atr_ma_50
        else:
            vol_regime = True  # Not enough data for regime filter yet
        
        # Breakout conditions: price closes outside Donchian channel
        bullish_breakout = curr_close > upper_channel
        bearish_breakout = curr_close < lower_channel
        
        # Exit conditions: reverse Donchian breakout or volatility collapse
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish Donchian breakout or volatility collapse
                if bearish_breakout or (i >= 50 and atr_value < 0.3 * atr_ma_50):
                    exit_signal = True
            elif position == -1:
                # Exit on bullish Donchian breakout or volatility collapse
                if bullish_breakout or (i >= 50 and atr_value < 0.3 * atr_ma_50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + volume spike + volatility regime
        if position == 0:
            # Long: break above upper Donchian channel
            long_condition = bullish_breakout and volume_spike and vol_regime
            # Short: break below lower Donchian channel
            short_condition = bearish_breakout and volume_spike and vol_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0