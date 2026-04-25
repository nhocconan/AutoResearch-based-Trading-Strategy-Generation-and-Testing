#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dVolumeSpike_ATRStop
Hypothesis: Donchian(20) breakout with 1d volume spike confirmation and ATR-based stoploss.
Works in bull markets via trend-following breaks and in bear markets via volatility expansion mean reversion.
Volume spike ensures institutional participation, ATR stop manages risk. Target: 20-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume spike and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR(14) for volatility filter and stoploss calculation
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
    tr1 = np.maximum(tr1, np.roll(df_1d['low'].values, 1))
    tr2 = np.abs(np.roll(df_1d['close'].values, 1) - df_1d['low'].values)
    tr3 = np.abs(np.roll(df_1d['close'].values, 1) - df_1d['high'].values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # first value has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 1d volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (2.0 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Donchian(20) channels on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop_multiplier = 2.5  # ATR multiplier for stoploss
    
    # Start index: need enough for 1d previous data (1) + 1d ATR (14) + volume MA (20) + Donchian (20)
    start_idx = max(14, 20, 20) + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals with all filters
            # Long breakout: price breaks above upper Donchian with volume spike
            long_breakout = (curr_close > high_20[i]) and vol_spike_aligned[i]
            # Short breakout: price breaks below lower Donchian with volume spike
            short_breakout = (curr_close < low_20[i]) and vol_spike_aligned[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below lower Donchian or ATR stoploss hit
            atr_stop = entry_price - (atr_stop_multiplier * atr_14_aligned[i])
            if curr_close < low_20[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above upper Donchian or ATR stoploss hit
            atr_stop = entry_price + (atr_stop_multiplier * atr_14_aligned[i])
            if curr_close > high_20[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dVolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0