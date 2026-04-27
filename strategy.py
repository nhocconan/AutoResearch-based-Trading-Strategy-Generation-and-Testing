#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Bollinger Band breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) with 1d EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below lower BB(20,2) with 1d EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses the 20-period SMA (middle band).
# Uses tight entry conditions to limit trades (~25/year) and avoid fee drag, targeting 100 total trades over 4 years.

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
    
    # Calculate Bollinger Bands (20,2)
    bb_period = 20
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    bb_up = np.full(n, np.nan)
    bb_dn = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        bb_mid[i] = np.mean(close[i - bb_period + 1:i + 1])
        bb_std[i] = np.std(close[i - bb_period + 1:i + 1])
        bb_up[i] = bb_mid[i] + 2 * bb_std[i]
        bb_dn[i] = bb_mid[i] - 2 * bb_std[i]
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB, EMA50, and volume MA20
    start_idx = max(bb_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_up[i]) or np.isnan(bb_dn[i]) or np.isnan(bb_mid[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above upper BB with 1d EMA50 uptrend and volume filter
            if (price > bb_up[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below lower BB with 1d EMA50 downtrend and volume filter
            elif (price < bb_dn[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle BB
            if price < bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle BB
            if price > bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger20_2_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0