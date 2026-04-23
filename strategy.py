#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price > upper Donchian(20) AND 1w EMA34 rising AND volume > 1.5x 20-period MA.
Short when price < lower Donchian(20) AND 1w EMA34 falling AND volume > 1.5x 20-period MA.
Exit when price crosses the 10-period EMA or opposite Donchian band is touched.
Uses 1w HTF for trend filter to avoid counter-trend trades in bear markets, volume spike for momentum confirmation.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    upper_donchian = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_donchian = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d EMA10 for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 10, 20)  # EMA34, Donchian, EMA10, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(ema_10[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1w_aligned[i-1]
            ema_rising = ema_34_1w_aligned[i] > ema_prev
            ema_falling = ema_34_1w_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 1.5x 20-period MA (strict threshold to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price > upper Donchian AND EMA34 rising AND volume filter
            if close[i] > upper_donchian[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian AND EMA34 falling AND volume filter
            elif close[i] < lower_donchian[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below EMA10 OR touches lower Donchian
                if close[i] < ema_10[i] or close[i] < lower_donchian[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above EMA10 OR touches upper Donchian
                if close[i] > ema_10[i] or close[i] > upper_donchian[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0