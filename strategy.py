#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-week trend filter and volume spike confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, weekly EMA50 for trend filter,
# and volume surge for entry confirmation. Designed to work in both bull and bear markets
# by capturing trend continuation after pullbacks. Target: 15-30 trades/year to avoid fee drag.
name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator components (13, 8, 5 periods with future shifts)
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need Alligator components
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price > all Alligator lines (bullish alignment) + weekly EMA50 up + volume spike
            if (price > jaw[i] and price > teeth[i] and price > lips[i] and 
                jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > ema_50_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < all Alligator lines (bearish alignment) + weekly EMA50 down + volume spike
            elif (price < jaw[i] and price < teeth[i] and price < lips[i] and 
                  jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                  close[i] < ema_50_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Jaw or weekly trend turns down
            if price < jaw[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Jaw or weekly trend turns up
            if price > jaw[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals