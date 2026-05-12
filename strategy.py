#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Volume Spike + Weekly Trend Filter
Hypothesis: Donchian channel breakouts capture strong trends. Weekly trend filter
avoids counter-trend trades. Volume spike confirms breakout strength. Designed
for very low trade frequency (~10-20/year) to minimize fee drag while capturing
sustained moves in both bull and bear markets.
"""
name = "1d_Donchian20_Breakout_VolumeSpike_WeeklyTrend"
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
    
    # === Donchian Channel (20-period) on 1d ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly Trend Filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume Spike (20-period on 1d) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + price above weekly EMA50 (uptrend)
            if (close[i] > donchian_high[i] and 
                vol_spike[i] and
                close[i] > ema_50_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + price below weekly EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_1d[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (reversal) OR close below weekly EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (reversal) OR close above weekly EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals