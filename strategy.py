#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + Daily Trend Filter
Hypothesis: Donchian(20) breakouts on 12h timeframe, when confirmed by volume spikes
and aligned with daily trend (price > daily EMA34 for longs, < for shorts), capture
strong directional moves. Designed for low trade frequency (~15-25/year) to minimize
fee decay while capturing sustained trends in both bull and bear markets.
"""
name = "12h_Donchian20_Breakout_Volume_DailyTrend"
timeframe = "12h"
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
    
    # === Donchian Channels (20-period on 12h) ===
    # Upper band: 20-period high, Lower band: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Daily EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (20-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure all indicators ready (max of 20, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > donchian_high[i] and 
                vol_spike[i] and
                close[i] > ema_34_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (reversal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (reversal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals