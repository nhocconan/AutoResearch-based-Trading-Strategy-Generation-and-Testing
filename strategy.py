#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike filter and ATR-based stoploss.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation (spike > 2.0x 20-period average).
- Donchian Channel(20): Upper = 20-period high, Lower = 20-period low.
- Entry: Long when close breaks above Upper AND volume confirmation.
         Short when close breaks below Lower AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits on lower break, short exits on upper break).
- Stoploss: ATR(20) * 2.0 from entry price (implemented as signal→0 when stop level breached on close).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by capturing breakouts with volume confirmation.
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
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(20) for stoploss on 12h timeframe
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Donchian Channel(20) on 12h timeframe
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, atr_period)
    
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
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average 1d volume
        # Note: Using 1d volume as proxy for institutional interest
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakout conditions
        breakout_upper = curr_close > curr_highest_high
        breakout_lower = curr_close < curr_lowest_low
        
        # Opposite breakout for exit
        breakout_lower_exit = curr_close < curr_lowest_low  # for long exit
        breakout_upper_exit = curr_close > curr_highest_high  # for short exit
        
        # Stoploss levels
        if position == 1:  # long position
            stop_loss_level = entry_price - 2.0 * curr_atr
            stop_loss_hit = curr_close < stop_loss_level
        elif position == -1:  # short position
            stop_loss_level = entry_price + 2.0 * curr_atr
            stop_loss_hit = curr_close > stop_loss_level
        else:
            stop_loss_hit = False
        
        # Exit conditions
        if position != 0:
            # Exit on opposite Donchian breakout OR stoploss hit
            if position == 1:
                if breakout_lower_exit or stop_loss_hit:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
            elif position == -1:
                if breakout_upper_exit or stop_loss_hit:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation
        if position == 0:
            # Long: breakout above upper channel AND volume confirmation
            long_condition = breakout_upper and volume_confirm
            
            # Short: breakout below lower channel AND volume confirmation
            short_condition = breakout_lower and volume_confirm
            
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
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dVolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0