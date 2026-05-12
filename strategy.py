#!/usr/bin/env python3
"""
12h Donchian Breakout + Daily Trend Filter + Volume Confirmation
Hypothesis: 12h timeframe captures medium-term trends with low trade frequency.
Breakouts above/below 20-period Donchian channels capture momentum.
Daily EMA50 filter ensures we only trade in the direction of the daily trend.
Volume confirmation reduces false breakouts. Designed for <30 trades/year
to minimize fee drift while capturing sustained moves in bull/bear markets.
"""
name = "12h_Donchian_Breakout_DailyTrend_Volume"
timeframe = "12h"
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
    
    # === 12h Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + daily EMA50 uptrend + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + daily EMA50 downtrend + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below lower Donchian OR daily trend turns down
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above upper Donchian OR daily trend turns up
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals