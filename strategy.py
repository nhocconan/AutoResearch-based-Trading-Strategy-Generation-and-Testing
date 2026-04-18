#!/usr/bin/env python3
"""
12h_1d_DonchianBreakout_Volume_TrendFilter
Hypothesis: On 12h timeframe, break above/below daily Donchian(20) high/low with volume > 1.5x 20-period average and trend filter (12h EMA50 > EMA200) captures strong momentum moves. Works in bull by riding breakouts above resistance, in bear by shorting breakdowns below support. Targets 15-25 trades/year by requiring confluence of breakout, volume, and trend. Uses daily Donchian for structure, avoiding whipsaws in chop.
"""

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
    
    # Get 1d data for Donchian channels (high/low of past 20 days)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian high and low on daily
    donch_high_20 = np.full(len(high_1d), np.nan)
    donch_low_20 = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high_20[i] = np.max(high_1d[i-20:i])
        donch_low_20[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 12h timeframe (wait for bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Trend filter: 12h EMA50 > EMA200 for long bias, < for short bias
    ema50 = np.full(n, np.nan)
    ema200 = np.full(n, np.nan)
    for i in range(50, n):
        ema50[i] = np.mean(close[i-50:i]) if not np.isnan(close[i-50:i]).any() else np.nan
    for i in range(200, n):
        ema200[i] = np.mean(close[i-200:i]) if not np.isnan(close[i-200:i]).any() else np.nan
    trend_up = ema50 > ema200
    trend_down = ema50 < ema200
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need EMA200 and Donchian warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above daily Donchian high, with volume, and uptrend
            if (close[i] > donch_high_aligned[i] and vol_confirm[i] and trend_up[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily Donchian low, with volume, and downtrend
            elif (close[i] < donch_low_aligned[i] and vol_confirm[i] and trend_down[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below daily Donchian low (failed breakout) or trend turns down
            if (close[i] < donch_low_aligned[i] or not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above daily Donchian high (failed breakdown) or trend turns up
            if (close[i] > donch_high_aligned[i] or not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_DonchianBreakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0