#!/usr/bin/env python3
# 12h_1d_alligator_2025
# Strategy: 12-hour Williams Alligator with 1-day trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Williams Alligator (three SMAs) identifies trends; when jaws (13-period) and teeth (8-period) are aligned and lips (5-period) crosses, it signals trend continuation. Combined with 1-day trend filter and volume confirmation to avoid false signals. Works in bull/bear by capturing trend continuations.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_alligator_2025"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator components (5, 8, 13 period SMAs on median price)
    median_price = (high + low) / 2.0
    
    # Calculate SMAs for Alligator
    # Jaws (13-period, 8 bars offset)
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period, 5 bars offset)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period, 3 bars offset)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day EMA to 12h timeframe (wait for daily close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.3 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Alligator signals: lips crossing teeth in direction of jaw alignment
        # Bullish: lips above teeth AND teeth above jaws (all aligned up)
        bullish_aligned = (lips[i] > teeth[i]) and (teeth[i] > jaws[i])
        # Bearish: lips below teeth AND teeth below jaws (all aligned down)
        bearish_aligned = (lips[i] < teeth[i]) and (teeth[i] < jaws[i])
        
        # Entry conditions: alignment + volume spike + trend filter
        long_signal = bullish_aligned and vol_spike[i] and uptrend_1d
        short_signal = bearish_aligned and vol_spike[i] and downtrend_1d
        
        # Exit when Alligator alignment breaks (lips crosses teeth opposite direction)
        exit_long = position == 1 and (lips[i] < teeth[i])  # Lips crossed below teeth
        exit_short = position == -1 and (lips[i] > teeth[i])  # Lips crossed above teeth
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals