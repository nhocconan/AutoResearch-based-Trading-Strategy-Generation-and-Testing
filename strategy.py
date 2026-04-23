#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR-based stop.
Long when price breaks above upper Donchian channel, close > 1d EMA34, and volume > 1.5x average.
Short when price breaks below lower Donchian channel, close < 1d EMA34, and volume > 1.5x average.
Exit via ATR trailing stop (3x ATR) or opposite Donchian breakout.
Designed for low trade frequency (~20-40/year) to capture strong trends with volume confirmation,
avoiding false breakouts in choppy markets. Works in both bull and bear markets by requiring
trend alignment via 1d EMA34.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for 4h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        atr_val = atr[i]
        close_val = close[i]
        ema34_val = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian, close > 1d EMA34, volume confirmation
            if (close_val > highest_high[i] and  # breakout above upper channel
                close_val > ema34_val and        # above 1d EMA34 (uptrend)
                vol_current > 1.5 * vol_ma_val): # volume confirmation
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            # Short: price breaks below lower Donchian, close < 1d EMA34, volume confirmation
            elif (close_val < lowest_low[i] and   # breakout below lower channel
                  close_val < ema34_val and       # below 1d EMA34 (downtrend)
                  vol_current > 1.5 * vol_ma_val): # volume confirmation
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, close_val)
                # Exit conditions: ATR trailing stop or opposite Donchian breakout
                if (close_val < highest_since_entry - 3.0 * atr_val or  # ATR stop
                    close_val < lowest_low[i]):                         # opposite breakout
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, close_val)
                # Exit conditions: ATR trailing stop or opposite Donchian breakout
                if (close_val > lowest_since_entry + 3.0 * atr_val or  # ATR stop
                    close_val > highest_high[i]):                       # opposite breakout
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0