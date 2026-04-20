#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w trend filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0, Bear Power < 0, and price > weekly EMA34 (uptrend)
# - Short when Bear Power < 0, Bull Power > 0, and price < weekly EMA34 (downtrend)
# - Volume must be > 1.5x 20-period average for confirmation
# - Exit when Elder Power signals reverse or price crosses weekly EMA34
# - Uses 1h for Elder Ray calculation (more responsive) and 1w for trend filter
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1h data for Elder Ray calculation (more responsive than 6h)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate EMA13 for Elder Ray (using 1h data)
    ema13_1h = pd.Series(close_1h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1h = high_1h - ema13_1h  # Bull Power = High - EMA13
    bear_power_1h = low_1h - ema13_1h   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1h, bull_power_1h)
    bear_power_6h = align_htf_to_ltf(prices, df_1h, bear_power_1h)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 6h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or np.isnan(ema34_1w_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, price > weekly EMA34, volume surge
            if (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and 
                price > ema34_1w_aligned[i] and vol > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Bear Power < 0, Bull Power > 0, price < weekly EMA34, volume surge
            elif (bear_power_6h[i] < 0 and bull_power_6h[i] > 0 and 
                  price < ema34_1w_aligned[i] and vol > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Elder Power signals reverse OR price crosses below weekly EMA34
            if (bull_power_6h[i] <= 0 or bear_power_6h[i] >= 0 or 
                price < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Elder Power signals reverse OR price crosses above weekly EMA34
            if (bear_power_6h[i] >= 0 or bull_power_6h[i] <= 0 or 
                price > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0