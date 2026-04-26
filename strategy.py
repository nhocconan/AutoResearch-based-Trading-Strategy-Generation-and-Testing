#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_Volume_ATRStop_v1
Hypothesis: On 4h timeframe, Donchian(20) breakouts with 12h EMA50 trend filter, volume confirmation, and ATR-based trailing stoploss produce high-quality trades in both bull and bear markets. The 12h trend filter ensures alignment with medium-term momentum, volume confirmation avoids false breakouts, and ATR stoploss manages risk. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for stoploss and volume MA
    tr1 = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])))
    tr2 = np.maximum(tr1, np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Start after warmup (need 50 for EMA, 20 for Donchian, 14 for ATR, 20 for volume MA)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(atr14[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter (EMA50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]
        breakout_down = close[i] < lowest_20[i]
        
        # Update trailing stops
        if position == 1:
            highest_since_long = max(highest_since_long, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if close[i] < highest_since_long - 2.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
                continue
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if close[i] > lowest_since_short + 2.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_short = 0.0
                continue
        
        # Long logic: breakout above Donchian high in uptrend with volume
        if uptrend and volume_spike and breakout_up:
            if position != 1:
                signals[i] = 0.25
                position = 1
                highest_since_long = high[i]  # initialize trailing stop
            else:
                signals[i] = 0.25
        # Short logic: breakout below Donchian low in downtrend with volume
        elif downtrend and volume_spike and breakout_down:
            if position != -1:
                signals[i] = -0.25
                position = -1
                lowest_since_short = low[i]  # initialize trailing stop
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0