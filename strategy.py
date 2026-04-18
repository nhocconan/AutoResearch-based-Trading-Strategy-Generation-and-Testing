#!/usr/bin/env python3
"""
12h_Turtle_Soup_v2
Strategy: 12h Turtle Soup pattern with 1D trend filter and volume confirmation.
Reduced trading frequency via stricter entry conditions:
- Requires both volume confirmation AND price action confirmation
- Uses hysteresis in trend filter to reduce whipsaw
- Implements minimum holding period to prevent churn
Designed for 12h timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull/bear via trend filter and mean-reversion breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 20-day high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 20-day high and low (Donchian channels)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 and EMA200 for trend filter with hysteresis
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    consecutive_bars = 0  # count bars in current position
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions with hysteresis to reduce whipsaw
        uptrend = ema_50_aligned[i] > ema_200_aligned[i] * 1.001  # 0.1% buffer
        downtrend = ema_50_aligned[i] < ema_200_aligned[i] * 0.999  # 0.1% buffer
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Turtle Soup conditions
        # Long: price tested below 20-day low, then closed back above it
        soup_long = low[i] < low_20_aligned[i] and close[i] > low_20_aligned[i]
        # Short: price tested above 20-day high, then closed back below it
        soup_short = high[i] > high_20_aligned[i] and close[i] < high_20_aligned[i]
        
        if position == 0:
            consecutive_bars = 0
            # Long: uptrend + volume + soup long setup
            if uptrend and vol_confirm and soup_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + soup short setup
            elif downtrend and vol_confirm and soup_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            consecutive_bars += 1
            # Minimum holding period: 2 bars to reduce churn
            if consecutive_bars < 2:
                signals[i] = 0.25
                continue
            
            # Long exit: trend change OR soup short setup (removed vol_confirm to reduce exits)
            if (not uptrend) or soup_short:
                signals[i] = -0.25  # reverse to short
                position = -1
                consecutive_bars = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            consecutive_bars += 1
            # Minimum holding period: 2 bars to reduce churn
            if consecutive_bars < 2:
                signals[i] = -0.25
                continue
            
            # Short exit: trend change OR soup long setup (removed vol_confirm to reduce exits)
            if (not downtrend) or soup_long:
                signals[i] = 0.25  # reverse to long
                position = 1
                consecutive_bars = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Turtle_Soup_v2"
timeframe = "12h"
leverage = 1.0