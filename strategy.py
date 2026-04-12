#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_Volume_Trend_v1
Hypothesis: On daily timeframe, price breaking above/below 20-day Donchian channel with 
volume confirmation and weekly trend filter (EMA21) captures strong trends in both bull and bear markets.
Weekly EMA21 ensures we only trade in the direction of the higher timeframe trend, reducing false breaks.
Volume surge (>2x 20-day average) confirms institutional interest. Designed for low trade frequency 
(10-25/year) to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Weekly EMA21 for trend filter
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Daily volume average (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema21_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only long if price above weekly EMA21, only short if below
        uptrend = close[i] > weekly_ema21_aligned[i]
        downtrend = close[i] < weekly_ema21_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = high[i] > donchian_high[i] and vol_spike[i] and uptrend
        short_breakout = low[i] < donchian_low[i] and vol_spike[i] and downtrend
        
        # Exit when price touches opposite Donchian band (mean reversion within channel)
        long_exit = low[i] <= donchian_low[i]
        short_exit = high[i] >= donchian_high[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals