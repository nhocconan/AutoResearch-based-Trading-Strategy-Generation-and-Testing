#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRStop_VolumeChop
Hypothesis: 4h Donchian(20) breakout with ATR-based stoploss and volatility regime filter.
Long when price breaks above 20-period Donchian high + ATR(14) < median ATR(50) (low volatility regime).
Short when price breaks below 20-period Donchian low + ATR(14) < median ATR(50).
ATR stoploss: exit long when price drops below highest high since entry - 2.5*ATR.
Exit short when price rises above lowest low since entry + 2.5*ATR.
Volume confirmation: current volume > 1.5 * 20-period average volume.
Designed for 75-200 total trades over 4 years (19-50/year) with discrete position sizing (±0.30).
Works in bull markets via breakouts and in bear markets via short breakdowns.
ATR stoploss manages risk during volatile regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate median ATR(50) for volatility regime filter
    atr_series = pd.Series(atr)
    median_atr_50 = atr_series.rolling(window=50, min_periods=50).median().values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Volatility filter: ATR(14) < median ATR(50) (low volatility regime)
    low_vol_regime = atr < median_atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Track highest high since entry for long stoploss
    highest_since_entry = np.full(n, np.nan)
    # Track lowest low since entry for short stoploss
    lowest_since_entry = np.full(n, np.nan)
    
    # Start after warmup
    start_idx = max(20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(median_atr_50[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Update highest high since entry for long positions
        if position == 1:
            if i == start_idx or position == 0:  # Just entered or continuing
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
        else:
            highest_since_entry[i] = np.nan
            
        # Update lowest low since entry for short positions
        if position == -1:
            if i == start_idx or position == 0:  # Just entered or continuing
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
        else:
            lowest_since_entry[i] = np.nan
        
        # Long logic: Donchian breakout + volume spike + low volatility regime
        if close[i] > donchian_high[i] and volume_spike[i] and low_vol_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
                highest_since_entry[i] = high[i]  # Reset tracking on new entry
            else:
                signals[i] = base_size
        # Short logic: Donchian breakdown + volume spike + low volatility regime
        elif close[i] < donchian_low[i] and volume_spike[i] and low_vol_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
                lowest_since_entry[i] = low[i]  # Reset tracking on new entry
            else:
                signals[i] = -base_size
        # Long stoploss: price drops below highest high since entry - 2.5*ATR
        elif position == 1 and not np.isnan(highest_since_entry[i]) and close[i] < (highest_since_entry[i] - 2.5 * atr[i]):
            signals[i] = 0.0
            position = 0
        # Short stoploss: price rises above lowest low since entry + 2.5*ATR
        elif position == -1 and not np.isnan(lowest_since_entry[i]) and close[i] > (lowest_since_entry[i] + 2.5 * atr[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_ATRStop_VolumeChop"
timeframe = "4h"
leverage = 1.0