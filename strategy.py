#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h EMA50 trend and volume surge
    # Works in bull markets (breakouts capture momentum) and bear markets (trend filter prevents counter-trend trades)
    # Donchian channels provide clear breakout levels, EMA50 filters trend direction, volume surge confirms strength
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 trend filter
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume surge AND 12h EMA50 uptrend
            if high[i] > high_max[i] and vol_surge[i] and close[i] > ema_12h_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band with volume surge AND 12h EMA50 downtrend
            elif low[i] < low_min[i] and vol_surge[i] and close[i] < ema_12h_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian middle or opposite band touch
            if position == 1:
                if low[i] < low_min[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if high[i] > high_max[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Trend_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0