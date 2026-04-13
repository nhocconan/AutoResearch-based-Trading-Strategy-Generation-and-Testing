#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and volume confirmation.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends via smoothed medians.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation adds conviction to signals.
# Designed to work in both bull and bear markets by following the trend.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Williams Alligator: Smoothed Medians (not SMA)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using Wilder's smoothing (EMA-like but for median price)
    median_price = (high + low) / 2
    
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan)
        if len(data) < period:
            return sma
        # First value is simple average
        sma[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_moving_average(median_price, 13)
    teeth = smoothed_moving_average(median_price, 8)
    lips = smoothed_moving_average(median_price, 5)
    
    # Align weekly trend filter: 50-period EMA of weekly close
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])  # Initialize with SMA
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 48) / 50  # EMA 50
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Warmup for Alligator
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend) + price above weekly EMA + volume
            if (lips_above_teeth and teeth_above_jaw and 
                price > weekly_ema and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Lips < Teeth < Jaw (downtrend) + price below weekly EMA + volume
            elif (lips_below_teeth and teeth_below_jaw and 
                  price < weekly_ema and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens (Lips < Teeth or Teeth < Jaw) OR volume drops
            if (not (lips_above_teeth and teeth_above_jaw) or 
                vol < 0.7 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakens (Lips > Teeth or Teeth > Jaw) OR volume drops
            if (not (lips_below_teeth and teeth_below_jaw) or 
                vol < 0.7 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Williams_Alligator_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0