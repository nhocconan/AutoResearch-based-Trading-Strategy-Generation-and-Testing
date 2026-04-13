#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13). 
# Long when Bull Power > 0 and rising + price > 1d EMA50 + volume > 1.5x avg.
# Short when Bear Power < 0 and falling + price < 1d EMA50 + volume > 1.5x avg.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray Index on 4h timeframe: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = np.zeros(n)
    ema_multiplier_13 = 2 / (13 + 1)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = (close[i] - ema13[i-1]) * ema_multiplier_13 + ema13[i-1]
    
    bull_power = high - ema13  # Bull Power
    bear_power = low - ema13   # Bear Power
    
    # Slope of Bull/Bear Power (3-period change)
    bull_power_slope = np.full(n, np.nan)
    bear_power_slope = np.full(n, np.nan)
    for i in range(3, n):
        bull_power_slope[i] = bull_power[i] - bull_power[i-3]
        bear_power_slope[i] = bear_power[i] - bear_power[i-3]
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema13[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Values
        bull_p = bull_power[i]
        bear_p = bear_power[i]
        bull_slope = bull_power_slope[i]
        bear_slope = bear_power_slope[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 and rising + above 1d EMA50 + volume confirmation
            if (bull_p > 0 and 
                bull_slope > 0 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 and falling + below 1d EMA50 + volume confirmation
            elif (bear_p < 0 and 
                  bear_slope < 0 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 or price breaks below 1d EMA50
            if (bull_p <= 0 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power >= 0 or price breaks above 1d EMA50
            if (bear_p >= 0 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ElderRay_Trend_Volume"
timeframe = "4h"
leverage = 1.0