#!/usr/bin/env python3
# 12h Donchian Breakout + Volume Spike + Daily Trend Filter
# Hypothesis: 12h Donchian breakouts capture medium-term trends in BTC/ETH/SOL.
# Volume spike confirms institutional participation.
# Daily EMA50 filter ensures alignment with longer-term trend, reducing whipsaw in choppy markets.
# Designed for low trade frequency (12-37/year) with clear entry/exit rules to minimize fee drag.

name = "12h_DonchianBreakout_Volume_DailyTrend"
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
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Donchian Channel (20-period) ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # === Volume Spike (20-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + volume spike + price above daily EMA50
            if (close[i] > donchian_upper[i] and 
                vol_spike[i] and
                close[i] > ema_50_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + volume spike + price below daily EMA50
            elif (close[i] < donchian_lower[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal signal)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal signal)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals