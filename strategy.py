#!/usr/bin/env python3
"""
12h_supertrend_volume_breakout_v1
Hypothesis: On 12h timeframe, use Supertrend (ATR-based trend filter) combined with Donchian channel breakouts and volume confirmation. Go long when price breaks above Donchian upper channel with Supertrend uptrend and volume > 1.5x average; go short when price breaks below Donchian lower channel with Supertrend downtrend and volume > 1.5x average. Exit when Supertrend reverses. This captures strong trending moves with volume confirmation while avoiding whipsaws in ranging markets. Works in both bull and bear via Supertrend's adaptive nature and volume confirmation. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_supertrend_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Supertrend
    atr_period = 10
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    atr[:atr_period] = np.nan
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3.0
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    uptrend = np.ones(n, dtype=bool)
    
    for i in range(n):
        if np.isnan(atr[i]):
            upper_band[i] = np.nan
            lower_band[i] = np.nan
            supertrend[i] = np.nan
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == 0:
            supertrend[i] = upper_band[i]
            uptrend[i] = True
        else:
            if close[i] <= upper_band[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            else:
                upper_band[i] = upper_band[i]
                
            if close[i] >= lower_band[i-1]:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            else:
                lower_band[i] = lower_band[i]
                
            if supertrend[i-1] == upper_band[i-1]:
                if close[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    uptrend[i] = True
                else:
                    supertrend[i] = lower_band[i]
                    uptrend[i] = False
            else:
                if close[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    uptrend[i] = False
                else:
                    supertrend[i] = upper_band[i]
                    uptrend[i] = True
    
    # Donchian channel (20-period)
    donchian_period = 20
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i < donchian_period:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-donchian_period:i])
            donchian_low[i] = np.min(low[i-donchian_period:i])
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 24:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(supertrend[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if Supertrend turns down
            if not uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if Supertrend turns up
            if uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with Supertrend uptrend and volume confirmation
            long_entry = False
            if (close[i] > donchian_high[i] and uptrend[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below Donchian low with Supertrend downtrend and volume confirmation
            short_entry = False
            if (close[i] < donchian_low[i] and not uptrend[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals