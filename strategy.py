#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy using 4h Donchian breakout with volume confirmation
# Uses 4h Donchian channels (20-period) for trend direction, entered on 1h breakouts
# Volume filter ensures institutional participation. Works in both bull and bear markets
# by requiring strong volume on breakouts. Targets 20-40 trades/year to minimize fee drag.

name = "1h_Donchian20_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-19:i+1])
        lower_4h[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian bands to 1h timeframe (use previous 4h bar's values)
    upper_4h_1h = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_1h = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_4h_1h[i]) or np.isnan(lower_4h_1h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian upper, above EMA50, volume spike
            if close[i] > upper_4h_1h[i] and close[i] > ema_50_1h[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian lower, below EMA50, volume spike
            elif close[i] < lower_4h_1h[i] and close[i] < ema_50_1h[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian lower or below EMA50
            if close[i] < lower_4h_1h[i] or close[i] < ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian upper or above EMA50
            if close[i] > upper_4h_1h[i] or close[i] > ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals