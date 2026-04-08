#!/usr/bin/env python3
# 4h_12h_donchian_breakout_volume_v1
# Hypothesis: Trade Donchian channel breakouts on 4h with 12h trend filter and volume confirmation.
# In bull markets, buy breakouts above upper band with 12h uptrend; in bear markets, sell breakdowns below lower band with 12h downtrend.
# Volume surge confirms breakout strength. Uses ATR-based stops to manage risk.
# Target: 20-50 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h trend: EMA25/50 crossover
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 4h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema25_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
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
            # Long entry: Break above upper Donchian band with 12h uptrend and volume surge
            if (high_max_20[i] > high_max_20[i-1] and  # New high breakout
                ema25_12h_aligned[i] > ema50_12h_aligned[i] and  # 12h uptrend
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below lower Donchian band with 12h downtrend and volume surge
            elif (low_min_20[i] < low_min_20[i-1] and  # New low breakdown
                  ema25_12h_aligned[i] < ema50_12h_aligned[i] and  # 12h downtrend
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals