#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Spike + ATR Stoploss
Hypothesis: Donchian breakouts capture sustained momentum. Combined with 1d EMA34 trend filter and volume confirmation,
this strategy targets 12h swings in both bull and bear markets. ATR-based stoploss manages risk. 12h timeframe targets 12-37 trades/year.
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for calculations
    start_idx = max(20, 34, 14)  # Donchian, daily EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Daily trend filter: price above/below EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND uptrend AND volume spike
            long_entry = (curr_high > highest_high[i]) and uptrend and vol_spike
            # Short: price breaks below Donchian lower AND downtrend AND volume spike
            short_entry = (curr_low < lowest_low[i]) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below Donchian lower OR stoploss hit
            if (curr_low < lowest_low[i]) or (curr_close < atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss upward
                atr_stop = max(atr_stop, curr_close - 2.5 * atr[i])
        elif position == -1:
            # Short position management
            # Exit: price breaks above Donchian upper OR stoploss hit
            if (curr_high > highest_high[i]) or (curr_close > atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss downward
                atr_stop = min(atr_stop, curr_close + 2.5 * atr[i])
    
    return signals

name = "12h_Donchian20_Breakout_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0