#!/usr/bin/env python3
# 1d Weekly Donchian Breakout + Volume Spike + 1w Trend Filter
# Hypothesis: Weekly Donchian breakouts capture major trend continuations, while volume confirms strength.
# The 1-week EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Works in bull markets (breakouts to new highs) and bear markets (breakdowns to new lows).
# Designed for very low trade frequency (~10-20/year) to minimize fee drag.
# Uses 1d as primary timeframe and 1w as HTF for trend filter.

name = "1d_WeeklyDonchianBreakout_Volume_Trend"
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
    
    # === Weekly Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(weekly_close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily Donchian Channel (20-period) ===
    # Upper band: highest high over last 20 days
    # Lower band: lowest low over last 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20-period on 1d) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + volume spike + price above weekly EMA50
            if (close[i] > donchian_upper[i] and 
                vol_spike[i] and
                close[i] > ema_50_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + volume spike + price below weekly EMA50
            elif (close[i] < donchian_lower[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_1d[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower (reversal signal)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper (reversal signal)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals