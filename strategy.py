#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend Filter and Volume Confirmation
Long: Price breaks above 20-day high + 1w EMA20 rising + volume > 1.5x average
Short: Price breaks below 20-day low + 1w EMA20 falling + volume > 1.5x average
Exit: Opposite breakout or volatility contraction
Designed to capture strong trends in both bull and bear markets with low trade frequency.
Target: 50-100 total trades over 4 years (12-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 20-day high/low and 20-week EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Get 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-week EMA on 1w data
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA20 to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate EMA20 slope (1-period change) for trend filter
    ema_20_slope = np.diff(ema_20_1w_aligned, prepend=ema_20_1w_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or
            np.isnan(avg_volume[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(ema_20_slope[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol = volume[i]
        donch_high = period20_high[i]
        donch_low = period20_low[i]
        vol_threshold = avg_volume[i] * 1.5
        ema_slope = ema_20_slope[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + 1w EMA20 rising + volume spike
            if price > donch_high and ema_slope > 0 and vol > vol_threshold:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + 1w EMA20 falling + volume spike
            elif price < donch_low and ema_slope < 0 and vol > vol_threshold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR volatility contraction
            if price < donch_low or vol < avg_volume[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR volatility contraction
            if price > donch_high or vol < avg_volume[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0