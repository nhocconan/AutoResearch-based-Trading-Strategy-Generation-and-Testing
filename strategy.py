#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1-week directional filter and volume confirmation
# - Long when price breaks above Donchian(20) high and 1-week EMA21 shows uptrend (price > EMA21)
# - Short when price breaks below Donchian(20) low and 1-week EMA21 shows downtrend (price < EMA21)
# - Volume must be > 1.5x 20-period average to confirm breakout strength
# - Weekly EMA21 provides strong trend filter to avoid counter-trend trades
# - Designed for 6h timeframe with selective breakout entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA21 calculation (weekly trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA21 on 1w timeframe
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA21 to 6h timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian channels on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average for confirmation
    volume_avg_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        volume = volume_6h[i]
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        ema21w = ema_21_1w_aligned[i]
        vol_avg = volume_avg_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume > 1.5 * vol_avg
        
        if position == 0:
            # Long entry: price breaks above Donchian high + weekly uptrend + volume confirmation
            if price > donchian_high and price > ema21w and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + weekly downtrend + volume confirmation
            elif price < donchian_low and price < ema21w and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly trend turns down
            if price < donchian_low or price < ema21w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly trend turns up
            if price > donchian_high or price > ema21w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_1wEMA21_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0