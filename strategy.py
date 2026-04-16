#!/usr/bin/env python3
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
    open_time = pd.DatetimeIndex(prices['open_time'])
    hours = open_time.hour
    
    # === 4h data (trend direction) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 1d data (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h EMA200 (trend filter) ===
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1d EMA50 (trend filter) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1h Donchian breakout (entry timing) ===
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema200_4h_val = ema200_4h_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        hh = highest_high[i]
        ll = lowest_low[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price breaks below 1h Donchian low OR trend turns bearish
            if price < ll or close[i] < ema200_4h_val or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price breaks above 1h Donchian high OR trend turns bullish
            if price > hh or close[i] > ema200_4h_val or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian high AND 4h/1d trend bullish
            if price > hh and close[i] > ema200_4h_val and close[i] > ema50_1d_val:
                signals[i] = 0.20
                position = 1
                continue
            # SHORT: price breaks below Donchian low AND 4h/1d trend bearish
            elif price < ll and close[i] < ema200_4h_val and close[i] < ema50_1d_val:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian_Trend_Filter_Session"
timeframe = "1h"
leverage = 1.0