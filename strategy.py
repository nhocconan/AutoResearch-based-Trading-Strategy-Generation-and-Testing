# 6h Donchian Breakout with Weekly Trend Filter
# Uses 6h Donchian channel breakout for entry, filtered by weekly trend direction
# Works in both bull/bear markets by only taking breakouts in the direction of weekly trend
# Volume confirmation reduces false breakouts
# Target: 15-30 trades/year (60-120 total over 4 years)

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly trend filter (HTF) ===
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # === 6h Donchian channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend: above EMA = uptrend, below EMA = downtrend
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Long entry: break above upper band in weekly uptrend
            if close[i] > highest_high[i] and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower band in weekly downtrend
            elif close[i] < lowest_low[i] and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals