#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Long when: Close breaks above 20-day Donchian high, weekly EMA50 up, volume > 1.5x 20-day avg.
# Short when: Close breaks below 20-day Donchian low, weekly EMA50 down, volume > 1.5x 20-day avg.
# Exit when price returns to Donchian midpoint or weekly trend reverses.
# Designed for low frequency (10-25 trades/year) with strong trend capture and volume confirmation.
# Weekly EMA filter ensures we only trade with the dominant trend, reducing whipsaws.
name = "1d_Donchian20_WeeklyTrend_Volume"
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
    
    # Daily Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Weekly trend filter: EMA(50) on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume vs 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        midpoint = donchian_mid[i]
        weekly_ema = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long: breakout above Donchian high, weekly EMA up, volume spike
            if price > highest and weekly_ema > ema_50_1w_aligned[i-1] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low, weekly EMA down, volume spike
            elif price < lowest and weekly_ema < ema_50_1w_aligned[i-1] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint or weekly EMA turns down
            if price < midpoint or weekly_ema < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or weekly EMA turns up
            if price > midpoint or weekly_ema > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals