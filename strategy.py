#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation
# In bull markets, price breaks above upper Donchian; in bear markets, breaks below lower Donchian
# Weekly trend filter prevents counter-trend trades during strong trends
# Volume confirms breakout conviction
# Works in both bull and bear by following the weekly trend direction
name = "1d_Donchian_WeeklyTrend_Volume"
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
    
    # Daily Donchian Channel (20-period)
    donch_period = 20
    upper_donch = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # Weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(vol_ema20[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + weekly uptrend
            if (price > upper_donch[i] and vol_confirm[i] and price > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + weekly downtrend
            elif (price < lower_donch[i] and vol_confirm[i] and price < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below lower Donchian
            if price < lower_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above upper Donchian
            if price > upper_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals