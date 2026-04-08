#!/usr/bin/env python3
# 4h_donchian_breakout_volume
# Hypothesis: Donchian(20) breakout with volume confirmation and ATR stoploss.
# Enters on breakout of 20-period high/low with volume > 1.5x average.
# Uses 12h EMA50 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag.
# Works in bull/bear markets by following trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter (EMA50) - load once before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period high/low)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: close below Donchian low OR ATR-based stop
            if close[i] < low_min[i] or close[i] < (high[i] - 2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above Donchian high OR ATR-based stop
            if close[i] > high_max[i] or close[i] > (low[i] + 2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long breakout: price closes above Donchian high in uptrend
                if uptrend and close[i] > high_max[i] and close[i-1] <= high_max[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below Donchian low in downtrend
                elif downtrend and close[i] < low_min[i] and close[i-1] >= low_min[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals