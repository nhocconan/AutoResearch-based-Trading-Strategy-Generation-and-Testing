#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h EMA(50) as trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when 12h EMA > price (bullish trend), price breaks above 4h Donchian upper band, volume > 1.5x average.
# Short when 12h EMA < price (bearish trend), price breaks below 4h Donchian lower band, volume > 1.5x average.
# Uses fixed position size of 0.25 to limit overtrading and fee drag. Target: 50-150 total trades over 4 years (12-38/year).
# Designed to work in bull (trend follow) and bear (trend still exists in downtrends) by using 12h trend filter.

name = "4h_12hEMA50_4hDonchian_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 12-hour EMA(50)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4-hour Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h EMA bullish (price > EMA), price breaks above 4h Donchian upper band, volume spike
            if (close[i] > ema_12h_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 12h EMA bearish (price < EMA), price breaks below 4h Donchian lower band, volume spike
            elif (close[i] < ema_12h_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend flip, price breaks below Donchian lower band, or max 20 bars held
            if (close[i] < ema_12h_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend flip, price breaks above Donchian upper band, or max 20 bars held
            if (close[i] > ema_12h_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals