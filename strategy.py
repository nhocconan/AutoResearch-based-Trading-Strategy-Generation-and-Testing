#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d trend filter and volume confirmation.
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In strong uptrends: Bull Power > 0 and rising. In strong downtrends: Bear Power < 0 and falling.
# Combined with 1d EMA trend filter (avoid counter-trend trades) and volume spikes (confirm conviction).
# Works in both bull and bear markets by using 1d trend filter to align with higher timeframe direction.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(20) for 1d trend filter
    ema20_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (20 + 1)
    ema20_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema20_1d[i] = (close_1d[i] - ema20_1d[i-1]) * ema_multiplier + ema20_1d[i-1]
    
    # Align 1d EMA to 6h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Elder Ray on 6h timeframe: EMA(13) for smoothing
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema_multiplier_13 = 2 / (13 + 1)
        ema13[0] = close[0]
        for i in range(1, n):
            ema13[i] = (close[i] - ema13[i-1]) * ema_multiplier_13 + ema13[i-1]
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema13[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1d_aligned[i]
        
        # Elder Ray values
        bp = bull_power[i]
        br = bear_power[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 and rising + above 1d EMA20 + volume confirmation
            if (bp > 0 and 
                i > 20 and bp > bull_power[i-1] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 and falling + below 1d EMA20 + volume confirmation
            elif (br < 0 and 
                  i > 20 and br < bear_power[i-1] and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or price breaks below 1d EMA
            if (bp <= 0 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive or price breaks above 1d EMA
            if (br >= 0 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0