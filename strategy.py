#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ATR-based stoploss.
# Long: Close breaks above Donchian upper AND volume > 2.0x 20-period MA
# Short: Close breaks below Donchian lower AND volume > 2.0x 20-period MA
# Exit: ATR trailing stop (highest high since entry - 2.5*ATR for longs, lowest low since entry + 2.5*ATR for shorts)
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian provides clear structure; volume confirmation reduces false breakouts.
# ATR stoploss adapts to volatility and works in both bull and bear markets.

name = "12h_Donchian20_VolumeSpike_ATR"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND volume spike
            if close_val > donchian_upper[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = high[i]
            # Short: Close breaks below Donchian lower AND volume spike
            elif close_val < donchian_lower[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # Long exit: ATR trailing stop (highest high since entry - 2.5*ATR)
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Short exit: ATR trailing stop (lowest low since entry + 2.5*ATR)
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals