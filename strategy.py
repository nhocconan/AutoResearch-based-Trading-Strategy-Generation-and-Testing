#!/usr/bin/env python3
# 1d_1w_DonchianBreakout_TrendFilter
# Hypothesis: Buy breakout above 20-day Donchian high in weekly uptrend, sell breakdown below 20-day low in weekly downtrend.
# Uses weekly EMA50 as trend filter to avoid counter-trend trades. Designed for low trade frequency (~10-25/year) to minimize fee drag.
# Works in bull/bear via weekly trend filter - only trades with the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_DonchianBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Daily Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and Donchian warmup
        # Get values
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema50_1w_val = ema50_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or np.isnan(ema50_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high in weekly uptrend
            if close_val > donchian_high_val and close_val > ema50_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in weekly downtrend
            elif close_val < donchian_low_val and close_val < ema50_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or weekly trend turns down
            if close_val < donchian_low_val or close_val < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or weekly trend turns up
            if close_val > donchian_high_val or close_val > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals