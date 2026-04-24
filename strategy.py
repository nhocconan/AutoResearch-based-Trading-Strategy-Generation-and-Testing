#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d data for weekly pivot levels and Donchian channels.
- Weekly pivot bias: price above weekly pivot (PP) = bullish bias, below = bearish bias.
- Entry: Long when price breaks above 6h Donchian upper (20) AND price > weekly PP AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below 6h Donchian lower (20) AND price < weekly PP AND volume > 1.5 * 6h volume MA(20).
- Exit: ATR-based trailing stop (2.5 * ATR(14)) from extreme since entry.
- Signal size: 0.25 discrete to control fee drag.
- Weekly pivot provides structural bias that works in both bull (buy dips above PP) and bear (sell rallies below PP) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and 6h Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # For each day, weekly pivot = (prior week's high + low + close) / 3
    # We need to group 1d data into weeks
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1)  # prior week's high
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1)   # prior week's low
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1)  # prior week's close
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp.values)
    
    # Calculate 6h Donchian channels (20-period)
    donch_hi_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lo_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 6h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14)  # Donchian needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_hi_6h[i]) or 
            np.isnan(donch_lo_6h[i]) or 
            np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(vol_ma_6h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry frequency
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above 6h Donchian high AND price > weekly PP (bullish bias)
                if curr_close > donch_hi_6h[i] and curr_close > weekly_pp_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Price breaks below 6h Donchian low AND price < weekly PP (bearish bias)
                elif curr_close < donch_lo_6h[i] and curr_close < weekly_pp_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.5 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotBias_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0