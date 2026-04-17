#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1-day volume confirmation and ADX trend filter.
In bull markets: price breaks above upper Donchian band (20-period high) with strong volume and ADX > 25.
In bear markets: price breaks below lower Donchian band (20-period low) with strong volume and ADX > 25.
This captures breakouts in both directions while filtering choppy markets. Volume confirms institutional participation.
Designed for low trade frequency (<50/year) to minimize fee drag.
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # ADX (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx = adx.values
    
    # 1-day volume confirmation (using 1d average volume)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_1d_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for Donchian(20) + ADX(14)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma_1d_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_20_aligned[i]
        
        if position == 0:
            # Long breakout: price > 20-period high + volume > 1.5x 1d avg + ADX > 25
            if price > highest_high[i] and vol > 1.5 * vol_ma and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < 20-period low + volume > 1.5x 1d avg + ADX > 25
            elif price < lowest_low[i] and vol > 1.5 * vol_ma and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price breaks below 10-period low (trailing exit) or ADX weak
            lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if price < lowest_low_10 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above 10-period high (trailing exit) or ADX weak
            highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if price > highest_high_10 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0