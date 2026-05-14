#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA(13): Bull = High - EMA13, Bear = Low - EMA13.
# In strong trends, power persists; in ranges, it oscillates around zero.
# Combined with 12h trend filter and volume spikes, it filters false signals and captures trends.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(20) for 12h trend filter
    ema20_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (20 + 1)
    ema20_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema20_12h[i] = (close_12h[i] - ema20_12h[i-1]) * ema_multiplier + ema20_12h[i-1]
    
    # Align 12h EMA to 6h timeframe
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Elder Ray on 6h timeframe
    # EMA(13) for power calculation
    ema13 = np.zeros(n)
    ema_multiplier_13 = 2 / (13 + 1)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = (close[i] - ema13[i-1]) * ema_multiplier_13 + ema13[i-1]
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Average volume (20-period = 20*6h = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_12h_aligned[i]
        
        # Elder Ray values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Strong bull power (> 0) + above 12h EMA20 + volume confirmation
            if (bull_val > 0 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Strong bear power (< 0) + below 12h EMA20 + volume confirmation
            elif (bear_val < 0 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull power turns negative or price breaks below 12h EMA
            if (bull_val <= 0 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear power turns positive or price breaks above 12h EMA
            if (bear_val >= 0 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0