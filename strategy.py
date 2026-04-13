#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR(14) stoploss.
# Long: Price breaks above Donchian upper channel (20-period high) + volume > 1.5x average volume (20-period).
# Short: Price breaks below Donchian lower channel (20-period low) + volume > 1.5x average volume.
# Exit: Reverse signal or ATR-based stoploss (close below highest high - 2*ATR for longs, above lowest low + 2*ATR for shorts).
# Uses 4h for signal generation with volume confirmation to filter false breakouts.
# ATR stoploss manages risk during adverse moves. Position size fixed at 0.25 to balance return and drawdown.
# Designed to work in both bull (breakouts) and bear (breakdowns) markets with volume filter reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = np.abs(high[i] - close[i-1])
        lc = np.abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above upper channel + volume confirmation
            if (price > upper and volume_confirm):
                position = 1
                entry_price = price
                signals[i] = position_size
            # Short: price breaks below lower channel + volume confirmation
            elif (price < lower and volume_confirm):
                position = -1
                entry_price = price
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions: reverse signal or ATR stoploss
            if (price < lower) or (price < entry_price - 2.0 * atr_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit conditions: reverse signal or ATR stoploss
            if (price > upper) or (price > entry_price + 2.0 * atr_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Volume_ATR_Stop"
timeframe = "4h"
leverage = 1.0