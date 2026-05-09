#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Breakouts capture momentum in trending markets; EMA filter ensures alignment with higher-timeframe trend.
# Volume confirmation reduces false breakouts. Works in both bull and bear markets by following the trend.
name = "4h_DonchianBreakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    donch_period = 20
    upper_channel = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_channel = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ema20[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper channel + above 12h EMA50 + volume confirmation
            if (price > upper_channel[i] and price > ema_50_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel + below 12h EMA50 + volume confirmation
            elif (price < lower_channel[i] and price < ema_50_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below lower channel
            if price < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above upper channel
            if price > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals