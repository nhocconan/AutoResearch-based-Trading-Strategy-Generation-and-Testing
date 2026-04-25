#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ATR Stoploss
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation ensures participation, and ATR-based stoploss manages risk. Works in bull (long on upside breakout) and bear (short on downside breakout). Target: 25-50 trades/year on 4h.
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
    
    # Get 1d data for ATR (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels (20) on 4h directly
    # We need to calculate on the primary timeframe (4h) since we're using 4h prices
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = high_max_20[i]
        donchian_low = low_min_20[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr_14_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high with volume confirmation
            long_entry = (curr_close > donchian_high) and volume_confirm
            # Short: price breaks below Donchian low with volume confirmation
            short_entry = (curr_close < donchian_low) and volume_confirm
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below Donchian low OR ATR-based stoploss
            atr_stop = entry_price - 2.5 * atr_val
            if curr_close < donchian_low or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price closes above Donchian high OR ATR-based stoploss
            atr_stop = entry_price + 2.5 * atr_val
            if curr_close > donchian_high or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0