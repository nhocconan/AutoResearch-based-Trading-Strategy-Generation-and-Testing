#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture momentum bursts. Volume confirms institutional participation.
ATR-based stoploss manages risk. Works in bull (upward breakouts) and bear (downward breakouts).
Designed for 4h timeframe with tight entry conditions (~30-50 trades/year) to minimize fee drag.
Uses discrete position sizing (0.0, ±0.30) for balance between return and risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need enough for Donchian, ATR, and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_donchian = high_ma[i]
        lower_donchian = low_ma[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND volume spike
            long_entry = (curr_high > upper_donchian) and vol_spike
            # Short: price breaks below lower Donchian AND volume spike
            short_entry = (curr_low < lower_donchian) and vol_spike
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_at_entry = atr_val
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: price drops below entry - 2.0 * ATR at entry
            stop_loss = entry_price - (2.0 * atr_at_entry)
            # Exit: price crosses below lower Donchian OR stoploss hit
            if (curr_close < lower_donchian) or (curr_close < stop_loss):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Stoploss: price rises above entry + 2.0 * ATR at entry
            stop_loss = entry_price + (2.0 * atr_at_entry)
            # Exit: price crosses above upper Donchian OR stoploss hit
            if (curr_close > upper_donchian) or (curr_close > stop_loss):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0