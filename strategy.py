#!/usr/bin/env python3
# 4h_12h_Camarilla_R1_S1_Breakout_Volume_Momentum
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h momentum filter (price > 12h EMA20) and volume confirmation.
# Enters long when price breaks above R1 in bullish momentum with volume surge, short when breaks below S1 in bearish momentum.
# Exits on close crossing the 5-period EMA in opposite direction.
# Designed for low trade frequency (15-30/year) to minimize fee drift and work in bull/bear markets.

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 12h data for momentum filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA20 for momentum filter
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # Using 1d data to get prior day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h 5-period EMA for exit
    ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_5[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum filter from 12h EMA20
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        bullish_momentum = close_12h_aligned[i] > ema_20_aligned[i]
        bearish_momentum = close_12h_aligned[i] < ema_20_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish momentum with volume
            if close[i] > r1_aligned[i] and bullish_momentum and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in bearish momentum with volume
            elif close[i] < s1_aligned[i] and bearish_momentum and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below 5-period EMA
                if close[i] < ema_5[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above 5-period EMA
                if close[i] > ema_5[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals