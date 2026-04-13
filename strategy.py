#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation.
# Uses weekly Donchian channels to identify primary trend, then enters on 12h breakouts
# in the direction of the weekly trend. Volume confirmation filters false breakouts.
# Designed to work in both bull (breakouts continue) and bear (mean reversion at extremes)
# markets by only trading in direction of higher timeframe trend.
# Target: 15-35 trades per year (60-140 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian(20) for trend identification
    donch_high_1w = np.full(len(high_1w), np.nan)
    donch_low_1w = np.full(len(low_1w), np.nan)
    for i in range(19, len(high_1w)):
        donch_high_1w[i] = np.max(high_1w[i-19:i+1])
        donch_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian to 12h timeframe (with 1-bar delay for completed weekly bar)
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # 12h Donchian(20) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Average volume (20-period = 10 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_high_1w_aligned[i]) or np.isnan(donch_low_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Weekly trend: price above weekly Donchian high = uptrend, below low = downtrend
        weekly_high = donch_high_1w_aligned[i]
        weekly_low = donch_low_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: 12h breakout above Donchian high + weekly uptrend + volume confirmation
            if (price > donch_high[i] and 
                price > weekly_high and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: 12h breakdown below Donchian low + weekly downtrend + volume confirmation
            elif (price < donch_low[i] and 
                  price < weekly_low and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 12h breakdown below Donchian low or weekly trend turns down
            if (price < donch_low[i] or price < weekly_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 12h breakout above Donchian high or weekly trend turns up
            if (price > donch_high[i] or price > weekly_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_Trend_Volume"
timeframe = "12h"
leverage = 1.0