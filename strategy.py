#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ATRStop_v1
Hypothesis: Donchian(20) breakout with volume spike confirmation and ATR-based stoploss. Designed to capture strong trending moves while avoiding choppy markets. Works in bull markets via upside breakouts and in bear markets via downside breakouts, with volume spike filter ensuring institutional participation. Targets low trade frequency (20-50/year) to minimize fee drag in ranging/bear markets like 2025+. Uses discrete position sizing (0.30) to control drawdown.
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of ATR, volume MA, Donchian
    start_idx = max(14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band AND volume spike
            long_signal = (close_val > highest_high[i]) and vol_spike
            
            # Short: price breaks below lower Donchian band AND volume spike
            short_signal = (close_val < lowest_low[i]) and vol_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price hits ATR stoploss (2.0 * ATR below entry) OR breaks below lower Donchian
            if (close_val < entry_price - 2.0 * atr[i]) or (close_val < lowest_low[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price hits ATR stoploss (2.0 * ATR above entry) OR breaks above upper Donchian
            if (close_val > entry_price + 2.0 * atr[i]) or (close_val > highest_high[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0