#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate daily EMA13 and EMA8 for Elder Ray
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema8_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema8_1d)
    
    # Volume confirmation: 20-period average on 6h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema8_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 and price > EMA8 (strong bullish momentum)
        if bull_power_aligned[i] > 0 and price_close > ema8_1d_aligned[i] and vol_confirm:
            enter_long = True
        
        # Short: Bear Power < 0 and price < EMA8 (strong bearish momentum)
        if bear_power_aligned[i] < 0 and price_close < ema8_1d_aligned[i] and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite signal or loss of momentum
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power becomes negative or price < EMA8
            exit_long = (bear_power_aligned[i] < 0) or (price_close < ema8_1d_aligned[i])
        elif position == -1:
            # Exit short if Bull Power becomes positive or price > EMA8
            exit_short = (bull_power_aligned[i] > 0) or (price_close > ema8_1d_aligned[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) with EMA8 filter and volume confirmation on 6h timeframe.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Measures strength relative to trend.
# Enters long when Bull Power > 0 and price above EMA8 (bullish momentum).
# Enters short when Bear Power < 0 and price below EMA8 (bearish momentum).
# Volume confirmation ensures institutional participation. Works in both bull and bear markets
# by capturing momentum shifts. Target: 50-150 total trades over 4 years.