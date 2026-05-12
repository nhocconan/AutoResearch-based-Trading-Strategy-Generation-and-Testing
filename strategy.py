#!/usr/bin/env python3
# 12h Donchian(20) Breakout + Daily Trend + Volume Spike
# Hypothesis: Breakouts from 20-period Donchian channels capture strong momentum moves.
# Combined with daily EMA trend filter and volume spikes for confirmation.
# Works in bull markets via upper band breaks and in bear markets via lower band breaks.
# Low trade frequency expected due to strict confluence requirements.

name = "12h_Donchian_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 20-period Donchian Channels ===
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + above daily EMA34 + volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_34_12h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + below daily EMA34 + volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_34_12h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower or trend change
            if close[i] < donchian_lower[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper or trend change
            if close[i] > donchian_upper[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals