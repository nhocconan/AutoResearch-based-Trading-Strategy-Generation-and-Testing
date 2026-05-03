#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based stoploss.
# Long when price breaks above Donchian upper channel (20-period high) with volume > 1.8x 20-period MA.
# Short when price breaks below Donchian lower channel (20-period low) with volume spike.
# Uses ATR(14) for dynamic stoploss: exit long if price drops 2.0*ATR below entry, exit short if price rises 2.0*ATR above entry.
# Designed for 4h timeframe to limit trades (target: 20-50/year) and work in both bull and bear markets via volatility breakouts.

name = "4h_Donchian20_VolumeSpike_ATRstop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # ATR(14) for stoploss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for ATR-based stoploss
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Donchian upper channel with volume spike
            if close_val > highest_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian lower channel with volume spike
            elif close_val < lowest_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below Donchian lower channel OR ATR stoploss hit
            if close_val < lowest_low[i] or close_val < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper channel OR ATR stoploss hit
            if close_val > highest_high[i] or close_val > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals