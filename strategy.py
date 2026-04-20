#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Channel Breakout with Volume Confirmation and Weekly Trend Filter
# - Uses weekly EMA21 for long-term trend direction (bull/bear filter)
# - Enters long when price breaks above 12h Donchian(20) high + volume > 1.5x 20-period average + weekly EMA21 up
# - Enters short when price breaks below 12h Donchian(20) low + volume > 1.5x 20-period average + weekly EMA21 down
# - Exits when price returns to Donchian midpoint or weekly trend reverses
# - Designed for 12h timeframe with tight entry conditions to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA21 for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_21 = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 12h Donchian Channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Volume confirmation: 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(weekly_ema_21[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        weekly_trend_up = weekly_ema_21[i] > weekly_ema_21[i-1] if i > 0 else weekly_ema_21[i] > 0
        weekly_trend_down = weekly_ema_21[i] < weekly_ema_21[i-1] if i > 0 else weekly_ema_21[i] < 0
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume confirmation + weekly uptrend
            if price > donchian_high and vol > vol_threshold[i] and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume confirmation + weekly downtrend
            elif price < donchian_low and vol > vol_threshold[i] and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or weekly trend turns down
            if price < donchian_mid[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or weekly trend turns up
            if price > donchian_mid[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_WeeklyEMA21_VolumeBreakout"
timeframe = "12h"
leverage = 1.0