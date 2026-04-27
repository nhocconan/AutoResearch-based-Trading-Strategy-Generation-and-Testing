#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator strategy with 12h trend filter and volume confirmation.
# Long when price > Alligator Teeth (Jaw+Teeth)/2 with 12h EMA50 uptrend and volume > 1.5x average.
# Short when price < Alligator Teeth with 12h EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses back below/above Teeth.
# Williams Alligator uses SMAs of 13, 8, 5 periods to identify trends. Works in trending markets.
# Williams Alligator + volume confirmation reduces false signals. Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Jaw SMA(13)
    jaw = np.full(n, np.nan)
    for i in range(jaw_period - 1, n):
        jaw[i] = np.mean(close[i - jaw_period + 1:i + 1])
    
    # Teeth SMA(8)
    teeth = np.full(n, np.nan)
    for i in range(teeth_period - 1, n):
        teeth[i] = np.mean(close[i - teeth_period + 1:i + 1])
    
    # Lips SMA(5) - not used directly but part of Alligator
    lips = np.full(n, np.nan)
    for i in range(lips_period - 1, n):
        lips[i] = np.mean(close[i - lips_period + 1:i + 1])
    
    # Alligator Teeth line: (Jaw + Teeth) / 2
    teeth_line = (jaw + teeth) / 2
    
    # Calculate 12-hour EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                          ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 12h indicators to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Jaw(13), Teeth(8), EMA50, and volume MA20
    start_idx = max(jaw_period - 1, teeth_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price > Teeth line with 12h EMA50 uptrend and volume
            if (price > teeth_line[i] and 
                price > ema_12h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price < Teeth line with 12h EMA50 downtrend and volume
            elif (price < teeth_line[i] and 
                  price < ema_12h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth line
            if price < teeth_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Teeth line
            if price > teeth_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_Teeth_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0