#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d volume spike and ATR volatility filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume spike confirmation.
- Donchian Channel: 20-period high/low breakout for trend following.
- Volume Filter: Current 12h volume > 2.0 * 20-period average 12h volume (avoid low-vol fakeouts).
- ATR Filter: Current ATR(14) < 2.0 * 20-period average ATR(14) to avoid extreme volatility whipsaws.
- Entry: Long when close > Donchian Upper AND volume confirmation AND ATR filter.
         Short when close < Donchian Lower AND volume confirmation AND ATR filter.
- Exit: Opposite Donchian break (long exits when close < Donchian Lower, short exits when close > Donchian Upper).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture sustained trends in both bull and bear markets while filtering chop/whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian Channel (20-period)
    # Using rolling window on primary timeframe data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) and its 20-period average for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for Donchian/volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(atr_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20[i]
        
        # ATR filter: current ATR < 2.0 * 20-period average ATR (avoid extreme volatility)
        atr_filter = curr_atr < 2.0 * atr_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper
        broke_below_lower = curr_close < lower
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below Donchian Lower
            if position == 1:
                if curr_close < lower:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above Donchian Upper
            elif position == -1:
                if curr_close > upper:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume and ATR filters
        if position == 0:
            # Long: break above Upper AND volume confirmation AND ATR filter
            long_condition = broke_above_upper and volume_confirm and atr_filter
            
            # Short: break below Lower AND volume confirmation AND ATR filter
            short_condition = broke_below_lower and volume_confirm and atr_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_12hVolSpike_ATRVolFilter_v1"
timeframe = "12h"
leverage = 1.0