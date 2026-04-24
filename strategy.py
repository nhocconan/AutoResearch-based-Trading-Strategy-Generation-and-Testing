#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume spike filter (current volume > 2.0 * 20-day average volume).
- Donchian Channel: Upper band = 20-period high, Lower band = 20-period low.
- Entry: Long when close breaks above upper band AND volume confirmation.
         Short when close breaks below lower band AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits when close < lower band, short exits when close > upper band).
- Stoploss: ATR(14) * 2.0 from entry price (implemented as signal=0 when stop level breached on close).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in trending markets by capturing breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for stoploss
    lookback_atr = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=lookback_atr, min_periods=lookback_atr).mean().values
    
    # Calculate Donchian Channel (20-period)
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_atr, lookback_dc)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0 * 20-day average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian levels
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        
        # Update ATR-based stoploss levels
        if position == 1:  # Long position
            long_stop = entry_price - 2.0 * atr[i]
        elif position == -1:  # Short position
            short_stop = entry_price + 2.0 * atr[i]
        
        # Check stoploss (using close price)
        if position != 0:
            if position == 1 and curr_close < long_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1 and curr_close > short_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions: Donchian breakout with volume confirmation
        if position == 0:
            # Long: close breaks above upper band AND volume confirmation
            long_condition = curr_close > upper_band and volume_confirm
            
            # Short: close breaks below lower band AND volume confirmation
            short_condition = curr_close < lower_band and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
            # Exit: close breaks below lower band (opposite Donchian)
            if curr_close < lower_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
            # Exit: close breaks above upper band (opposite Donchian)
            if curr_close > upper_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0