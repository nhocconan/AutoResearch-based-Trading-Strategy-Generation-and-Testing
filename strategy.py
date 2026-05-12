#!/usr/bin/env python3
# 1d_Donchian20_1wTrend_Volume
# Hypothesis: Buy when price breaks above 20-day Donchian high with weekly uptrend and volume confirmation.
# Sell when price breaks below 20-day Donchian low with weekly downtrend and volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Designed for low frequency (10-25 trades/year) to capture major trends in BTC/ETH while minimizing fee drag.

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # === Weekly trend filter: EMA34 on weekly close ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Donchian channels (20-period) ===
    # Highest high of last 20 days
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 days
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-day average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high, weekly uptrend, volume confirmation
            if close[i] > donchian_high[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, weekly downtrend, volume confirmation
            elif close[i] < donchian_low[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or weekly trend turns down
            if close[i] < donchian_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or weekly trend turns up
            if close[i] > donchian_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals