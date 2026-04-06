#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR-based Donchian breakout with volume confirmation
# Long when price breaks above Donchian(10) high + volume > 1.5x 20-period average
# Short when price breaks below Donchian(10) low + volume > 1.5x 20-period average
# Uses ATR(14) for dynamic stoploss and position sizing
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range

name = "12h_atr_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    atr[14] = np.mean(tr[:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume average (20-period)
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = vol_sum / (i+1) if i > 0 else volume[0]
    
    # Donchian channels (10-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(9, n):
        donch_high[i] = np.max(high[i-9:i+1])
        donch_low[i] = np.min(low[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = volume[i] > 1.5 * vol_ma[i]
        
        # Check exits: stoploss or reversal
        if position == 1:  # long position
            # Exit: price closes below Donchian low OR stoploss hit (2*ATR)
            if close[i] <= donch_low[i] or close[i] <= (entry_price - 2 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high OR stoploss hit (2*ATR)
            if close[i] >= donch_high[i] or close[i] >= (entry_price + 2 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above Donchian high + volume confirmation
            if close[i] > donch_high[i] and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume confirmation
            elif close[i] < donch_low[i] and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals