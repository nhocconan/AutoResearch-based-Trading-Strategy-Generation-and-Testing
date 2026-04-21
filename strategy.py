#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1w EMA100 trend filter and volume confirmation.
Trend filter: price > 1w EMA100 for long, price < 1w EMA100 for short.
Breakout: price breaks above/below 12h Donchian(20) channel.
Volume confirmation: volume > 1.5x 20-period average.
Exit: opposite breakout or trend reversal.
Designed for 12h to target 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA100 for trend filter
    close_1w = df_1w['close'].values
    ema_100 = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1w, ema_100)
    
    # Load 12h data ONCE before loop for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h Donchian(20) channel
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_100_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_100_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian high + uptrend + volume spike
            if (price_close > donch_high and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian low + downtrend + volume spike
            elif (price_close < donch_low and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite breakout or trend reversal
            if position == 1:
                if price_close < donch_low or price_close < ema_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > donch_high or price_close > ema_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA100_Volume"
timeframe = "12h"
leverage = 1.0