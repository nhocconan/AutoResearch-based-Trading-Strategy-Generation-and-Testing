#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (momentum) with 1d EMA50 uptrend and volume > 1.5x average.
# Short when Bear Power < 0 and Bull Power falling with 1d EMA50 downtrend and volume > 1.5x average.
# Exit when Elder Ray signal reverses (Bull Power < 0 for long exit, Bear Power > 0 for short exit).
# Uses Elder Ray to measure bull/bear power relative to EMA, targeting 12-37 trades per year on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate EMA13 for Elder Ray
    ema13_period = 13
    ema13 = np.full(n, np.nan)
    if n >= ema13_period:
        ema13[ema13_period - 1] = np.mean(close[:ema13_period])
        for i in range(ema13_period, n):
            ema13[i] = (close[i] * (2 / (ema13_period + 1)) + 
                        ema13[i - 1] * (1 - (2 / (ema13_period + 1))))
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Calculate Elder Ray momentum (change in power)
    bull_power_mom = np.full(n, np.nan)
    bear_power_mom = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(bull_power[i]) and not np.isnan(bull_power[i-1]):
            bull_power_mom[i] = bull_power[i] - bull_power[i-1]
        if not np.isnan(bear_power[i]) and not np.isnan(bear_power[i-1]):
            bear_power_mom[i] = bear_power[i] - bear_power[i-1]
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA13, Elder Ray, EMA50, and volume MA20
    start_idx = max(ema13_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_mom[i]) or np.isnan(bear_power_mom[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, with 1d EMA50 uptrend and volume filter
            if (bull_power[i] > 0 and bull_power_mom[i] > 0 and 
                bear_power[i] < 0 and price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0, with 1d EMA50 downtrend and volume filter
            elif (bear_power[i] < 0 and bear_power_mom[i] < 0 and 
                  bull_power[i] > 0 and price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power >= 0 (loss of bearish momentum)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0