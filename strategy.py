#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# In trending markets (price > 1d EMA50), break above upper channel or below lower channel with volume triggers continuation entries.
# In ranging markets (price near 1d EMA50), fade at Donchian extremes for mean reversion.
# Uses ATR-based trailing stop (2.5x) to manage risk. Designed for low trade frequency (~12-37/year) to minimize fee drag.
# Works in bull/bear via regime adaptation: trend following in strong trends, mean reversion in ranges.

name = "12h_Donchian20_1dEMA50_RegimeAdaptive_VolumeSpike_ATRTrail_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian channels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower channels
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Donchian channels and EMA50 to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h ATR(14) for dynamic trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_upper = upper_aligned[i]
        curr_lower = lower_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long breakouts above upper channel with volume
                if curr_close > curr_upper and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
            elif is_downtrend:
                # In downtrend: look for short breakdowns below lower channel with volume
                if curr_close < curr_lower and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
            else:
                # In ranging market (near EMA): mean reversion at Donchian extremes
                if curr_close < curr_lower:
                    # Near lower channel: look for long
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                elif curr_close > curr_upper:
                    # Near upper channel: look for short
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals