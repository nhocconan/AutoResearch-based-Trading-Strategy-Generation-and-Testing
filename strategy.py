#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend and Volume Confirmation
Hypothesis: Donchian channel breakouts on 12h timeframe capture sustained moves,
filtered by daily trend and volume spikes to avoid false breakouts. Designed for
low trade frequency (target: 12-37 trades/year) to minimize fee drift in both
bull and bear markets.
"""
name = "12h_Donchian_Breakout_DailyTrend_Volume"
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
    
    # === DAILY TREND (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 12H DONCHIAN CHANNEL (20) ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 12H VOLUME (20) SPIKE ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, donchian_period, 20)  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_1d[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Above daily EMA + break above Donchian high + volume spike
            if (close[i] > trend_1d[i] and 
                close[i] > donchian_high[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Below daily EMA + break below Donchian low + volume spike
            elif (close[i] < trend_1d[i] and 
                  close[i] < donchian_low[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below Donchian low OR volume dries up
            if close[i] < donchian_low[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high OR volume dries up
            if close[i] > donchian_high[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals