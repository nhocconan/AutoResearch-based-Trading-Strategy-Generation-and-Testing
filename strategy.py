#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation.
Long when price breaks above 12h Donchian high (20) and 1-day EMA34 trend is up and volume > 1-day average.
Short when price breaks below 12h Donchian low (20) and 1-day EMA34 trend is down and volume > 1-day average.
Exit when price crosses 12h Donchian midline or trend reverses.
Donchian provides clear breakout levels; EMA34 filter ensures trading with trend; volume confirms institutional interest.
Works in both bull and bear markets by filtering for strong trends and avoiding chop.
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
    
    # Load 1-day data for trend and volume filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1-day EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_trend_up = ema_34_1d > ema_34_1d_prev
    ema_trend_down = ema_34_1d < ema_34_1d_prev
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_up)
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_down)
    
    # 1-day average volume for confirmation
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 12-hour Donchian channels (20 periods)
    donch_len = 20
    highest_high = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lowest_low = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    donch_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high, uptrend, volume confirmation
            if (close[i] > highest_high[i] and 
                ema_trend_up_aligned[i] and 
                volume[i] > avg_vol_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low, downtrend, volume confirmation
            elif (close[i] < lowest_low[i] and 
                  ema_trend_down_aligned[i] and 
                  volume[i] > avg_vol_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian midline OR trend turns down
                if (close[i] < donch_mid[i] or not ema_trend_up_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian midline OR trend turns up
                if (close[i] > donch_mid[i] or not ema_trend_down_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0