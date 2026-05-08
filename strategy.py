#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + weekly close > weekly open + volume spike
# Short when price breaks below Donchian(20) low + weekly close < weekly open + volume spike
# Exit when price crosses back through Donchian midpoint or weekly trend reverses
# Weekly trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation in breakout
# Targets 30-100 total trades over 4 years (7-25/year) to avoid fee drag

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly bullish/bearish based on close vs open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True when weekly close > open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate Donchian midpoint for exit
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback - 1  # start when Donchian is fully calculated
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        vol_spike = volume_spike[i]
        don_high = highest_high[i]
        don_low = lowest_low[i]
        don_mid = donchian_mid[i]
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + weekly bullish + volume spike
            if price > don_high and weekly_bull and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + weekly bearish + volume spike
            elif price < don_low and not weekly_bull and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR weekly turns bearish
            if price < don_mid or not weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR weekly turns bullish
            if price > don_mid or weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals