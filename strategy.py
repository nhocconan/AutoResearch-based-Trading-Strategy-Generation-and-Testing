#!/usr/bin/env python3
# 12h_1d_donchian_breakout_volume_v2
# Hypothesis: Trade 12-hour Donchian channel breakouts with 1-day trend filter and volume confirmation.
# In bull markets, buy breakouts above upper band with 1-day uptrend; in bear markets, sell breakdowns below lower band with 1-day downtrend.
# Volume surge confirms breakout strength. Uses ATR-based stops to manage risk.
# Target: 12-37 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day trend: EMA25/50 crossover
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 12h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema25_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below lower Donchian band OR stoploss hit
            if close[i] < low_min_20[i] or close[i] < high_max_20[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper Donchian band OR stoploss hit
            if close[i] > high_max_20[i] or close[i] > low_min_20[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above upper Donchian band with 1-day uptrend and volume surge
            if (high_max_20[i] > high_max_20[i-1] and  # New high breakout
                ema25_1d_aligned[i] > ema50_1d_aligned[i] and  # 1-day uptrend
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below lower Donchian band with 1-day downtrend and volume surge
            elif (low_min_20[i] < low_min_20[i-1] and  # New low breakdown
                  ema25_1d_aligned[i] < ema50_1d_aligned[i] and  # 1-day downtrend
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals