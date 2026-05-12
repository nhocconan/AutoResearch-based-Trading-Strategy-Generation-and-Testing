#!/usr/bin/env python3
# 4h Donchian Breakout + 1d Trend + Volume Spike
# Hypothesis: Combines price breakout (Donchian channels) with 1d trend filter and volume confirmation.
# Donchian breakout captures momentum; 1d EMA filter ensures alignment with higher timeframe trend;
# volume spike validates breakout strength. Works in bull markets via long breakouts and
# bear markets via short breakouts. Designed for low trade frequency (<50/year) to minimize fee drag.

name = "4h_Donchian_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Donchian Channels (20-period) ===
    # Upper band: highest high over past 20 periods
    # Lower band: lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Ensure all indicators ready (20 for Donchian + 34 for EMA + buffer)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above Donchian high + price above 1d EMA34 + volume spike
            if close[i] > donchian_high[i] and close[i] > ema_34_4h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Donchian low + price below 1d EMA34 + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_34_4h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close breaks below Donchian low OR trend reversal
            if close[i] < donchian_low[i] or close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above Donchian high OR trend reversal
            if close[i] > donchian_high[i] or close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals