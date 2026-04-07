#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume and 1w Trend Filter
Long when price breaks above Donchian upper band with above-average volume and weekly trend up
Short when price breaks below Donchian lower band with above-average volume and weekly trend down
Exit when price crosses opposite Donchian band or volume drops
Designed for fewer trades (target 15-30/year) to avoid fee drag, works in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_1w_trend_v1"
timeframe = "12h"
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
    
    # === Donchian Channels (20-period) ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # === Weekly trend filter (EMA 21 on weekly close) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_trend = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(weekly_trend[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below lower band OR volume drops below average
            if close[i] < lowest_low[i] or vol_ratio[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band OR volume drops below average
            if close[i] > highest_high[i] or vol_ratio[i] < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume and weekly trend confirmation
            if close[i] > highest_high[i] and weekly_trend[i] > weekly_close[-1] if len(weekly_close) > 0 else False:
                # Price above upper band and weekly trend up -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and weekly_trend[i] < weekly_close[-1] if len(weekly_close) > 0 else True:
                # Price below lower band and weekly trend down -> short
                position = -1
                signals[i] = -0.25
    
    return signals