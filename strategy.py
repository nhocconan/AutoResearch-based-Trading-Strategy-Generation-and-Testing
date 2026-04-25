#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1d EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
In bull markets: price breaks above upper channel with EMA50 uptrend and volume spike → long.
In bear markets: price breaks below lower channel with EMA50 downtrend and volume spike → short.
Uses 1d HTF for EMA50 trend filter. Volume confirmation reduces false breakouts.
ATR-based stoploss manages risk. Designed for 12h timeframe to limit trade frequency (target: 12-37/year).
Works in both bull and bear markets by following the trend direction from higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss calculation
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) - calculated on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, EMA, and volume MA
    start_idx = max(20, 50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 1d EMA50 trend alignment
            breakout_long = curr_close > highest_high[i-1]  # break above previous period's high
            breakout_short = curr_close < lowest_low[i-1]   # break below previous period's low
            
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Donchian lower band retrace or stoploss
            donchian_exit = curr_close < lowest_low[i]
            stoploss_hit = curr_close < entry_price - (2.5 * atr_14[i])
            
            if donchian_exit or stoploss_hit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit on Donchian upper band retrace or stoploss
            donchian_exit = curr_close > highest_high[i]
            stoploss_hit = curr_close > entry_price + (2.5 * atr_14[i])
            
            if donchian_exit or stoploss_hit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0