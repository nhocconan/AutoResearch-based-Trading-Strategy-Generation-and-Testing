#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator with 6h EMA8/34 crossover and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA of median price) AND EMA8 > EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price < Alligator Jaw AND EMA8 < EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when EMA8/EMA34 crossover reverses
# Uses discrete sizing 0.25 to control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams Alligator identifies trending vs ranging markets via smoothed medians
# EMA8/34 provides timely entry signals within Alligator-defined trends
# Volume confirmation ensures breakout validity
# Works in bull (price above Jaw in uptrend) and bear (price below Jaw in downtrend)

name = "6h_1dWilliamsAlligator_6hEMA8_34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for Alligator (13+8+5 periods)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (SMMA of median price)
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan, dtype=np.float64)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA of median price
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Get 6h data ONCE before loop for EMA8/34
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 40:  # Need sufficient data for EMA34
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA8 and EMA34
    close_series_6h = pd.Series(close_6h)
    ema_8_6h = close_series_6h.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_34_6h = close_series_6h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Alligator lines to 6h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Align 6h EMA indicators to 6h timeframe (wait for completed 6h bar)
    ema_8_aligned = align_htf_to_ltf(prices, df_6h, ema_8_6h)
    ema_34_aligned = align_htf_to_ltf(prices, df_6h, ema_34_6h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema_8_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Alligator Jaw AND EMA8 > EMA34 AND volume confirmation
            if (close[i] > jaw_aligned[i] and 
                ema_8_aligned[i] > ema_34_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Alligator Jaw AND EMA8 < EMA34 AND volume confirmation
            elif (close[i] < jaw_aligned[i] and 
                  ema_8_aligned[i] < ema_34_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA8/EMA34 crossover reverses (EMA8 < EMA34)
            if ema_8_aligned[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA8/EMA34 crossover reverses (EMA8 > EMA34)
            if ema_8_aligned[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals