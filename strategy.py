#/usr/bin/env python3
# 4h Donchian Breakout with Volume Confirmation and Daily EMA Trend Filter
# Hypothesis: Donchian channel breakouts capture momentum, validated by volume spikes and aligned with daily trend.
# Works in both bull and bear markets by only taking breakouts in the direction of the daily EMA50 trend.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "4h_Donchian_Breakout_Volume_DailyTrend"
timeframe = "4h"
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
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above upper Donchian + volume spike + price above daily EMA50
            if (close[i] > highest_high[i] and 
                vol_spike[i] and
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian + volume spike + price below daily EMA50
            elif (close[i] < lowest_low[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price re-enters the Donchian channel (below midpoint)
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters the Donchian channel (above midpoint)
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals