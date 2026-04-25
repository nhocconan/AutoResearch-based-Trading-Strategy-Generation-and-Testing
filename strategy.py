#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d ATR Regime Filter and Volume Spike Confirmation
Hypothesis: Donchian(20) breakouts capture strong momentum moves. 1d ATR regime filter adapts to volatility - 
in high volatility (ATR > 20-bar MA), we trade breakouts; in low volatility, we avoid false signals. 
Volume spike (>1.5x 20-bar vol MA) confirms momentum. Designed for both bull and bear markets by 
trading breakouts in direction of prevailing trend. Target: 25-35 trades/year to minimize fee drag.
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
    
    # Get 1d data for ATR regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need 20 for ATR calculation
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ATR MA(20) for regime threshold
    atr_ma_20_1d = np.full(len(atr_14_1d_aligned), np.nan)
    for i in range(20, len(atr_14_1d_aligned)):
        atr_ma_20_1d[i] = np.mean(atr_14_1d_aligned[i-19:i+1])
    
    # Donchian channels on 4h data (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, ATR regime, and volume MA
    start_idx = max(40, 20)  # 40 for ATR regime (20+20), 20 for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_ma_20_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_14_val = atr_14_1d_aligned[i]
        atr_ma_val = atr_ma_20_1d[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ma = vol_ma_20[i]
        
        # ATR regime filter: trade only in high volatility regimes
        high_volatility = atr_14_val > atr_ma_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            if high_volatility and volume_confirm:
                # Long breakout: price breaks above Donchian high
                long_signal = curr_close > donch_high
                # Short breakout: price breaks below Donchian low
                short_signal = curr_close < donch_low
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to midpoint OR volatility drops
            midpoint = (donch_high + donch_low) / 2
            if curr_close < midpoint or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to midpoint OR volatility drops
            midpoint = (donch_high + donch_low) / 2
            if curr_close > midpoint or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0