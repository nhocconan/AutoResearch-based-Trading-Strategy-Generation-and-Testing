#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_Trend_v1
Hypothesis: On 12h timeframe, buy when price breaks above 20-period Donchian high with daily uptrend (close > SMA50),
sell when price breaks below 20-period Donchian low with daily downtrend (close < SMA50). Exit when price crosses the
midpoint of the Donchian channel. Uses volume confirmation (>1.5x average volume) to avoid false breakouts.
Designed for low trade frequency (15-30/year) by requiring multiple confluence factors. Works in bull/bear via daily
trend filter and mean-reversion exit at midpoint.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY TREND FILTER (SMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_50 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < 49:
            sma_50[i] = np.nan
        else:
            sma_50[i] = np.mean(close_1d[i-49:i+1])
    
    # Trend: 1 for uptrend (close > SMA50), -1 for downtrend (close < SMA50), 0 otherwise
    trend_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if not np.isnan(sma_50[i]):
            if close_1d[i] > sma_50[i]:
                trend_1d[i] = 1
            elif close_1d[i] < sma_50[i]:
                trend_1d[i] = -1
    
    # === 12H DONCHIAN CHANNEL (20-period) ===
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(n):
        if i < donchian_period - 1:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            start_idx = i - donchian_period + 1
            donchian_high[i] = np.max(high[start_idx:i+1])
            donchian_low[i] = np.min(low[start_idx:i+1])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # === VOLUME AVERAGE (20-period) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align daily trend to 12h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(trend_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions
        long_setup = (close[i] > donchian_high[i]) and (trend_aligned[i] > 0.5) and vol_confirm
        short_setup = (close[i] < donchian_low[i]) and (trend_aligned[i] < -0.5) and vol_confirm
        
        # Exit conditions: mean reversion to midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals