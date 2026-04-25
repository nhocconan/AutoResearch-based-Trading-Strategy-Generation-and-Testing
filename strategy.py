#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d ATR-based volatility filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR calculation and trend context.
- Donchian Breakout: Upper/lower bands from 20-period high/low on 12h timeframe.
- Volatility Filter: Current 12h ATR > 1.5 * 20-period average 12h ATR to confirm momentum burst.
- Volume Filter: Current 12h volume > 2.0 * 20-period average 12h volume.
- Entry: Long when close > Upper Band AND volatility filter AND volume filter.
         Short when close < Lower Band AND volatility filter AND volume filter.
- Exit: Opposite Donchian break (long exits when close < Lower Band, short exits when close > Upper Band).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts during high volatility periods while filtering low-volatility chop.
- Works in bull markets (trend continuation) and bear markets (trend continuation down).
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
    
    # Calculate 12h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR average for volatility filter (20-period)
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for ATR (14+20), 20 for Donchian/MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        atr_value = atr[i]
        atr_ma = atr_ma_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility filter: current ATR > 1.5 * 20-period average ATR
        volatility_filter = atr_value > 1.5 * atr_ma
        
        # Volume filter: current volume > 2.0 * 20-period average volume
        volume_filter = curr_volume > 2.0 * vol_ma
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_band
        broke_below_lower = curr_close < lower_band
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower band
            if position == 1:
                if curr_close < lower_band:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper band
            elif position == -1:
                if curr_close > upper_band:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility and volume filters
        if position == 0:
            # Long: break above upper band AND volatility filter AND volume filter
            long_condition = broke_above_upper and volatility_filter and volume_filter
            
            # Short: break below lower band AND volatility filter AND volume filter
            short_condition = broke_below_lower and volatility_filter and volume_filter
            
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

name = "12h_Donchian20_Breakout_VolATR_Filter_v1"
timeframe = "12h"
leverage = 1.0