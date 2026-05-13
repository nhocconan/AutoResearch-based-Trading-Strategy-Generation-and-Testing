#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_Trend_Force_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly Donchian Channel (20-period lookback)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_period, n):
        upper[i] = np.max(high[i-donchian_period:i])
        lower[i] = np.min(low[i-donchian_period:i])
    
    # Weekly trend filter: 20-period EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above upper Donchian with weekly uptrend and volume spike
            if close[i] > upper[i] and close[i] > ema20_1w_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian with weekly downtrend and volume spike
            elif close[i] < lower[i] and close[i] < ema20_1w_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower Donchian
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper Donchian
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals