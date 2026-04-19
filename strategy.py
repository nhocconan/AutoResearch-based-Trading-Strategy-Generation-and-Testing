#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
# Long when: Price breaks above 20-day high, weekly EMA50 upward, volume > 1.5x 20-day average
# Short when: Price breaks below 20-day low, weekly EMA50 downward, volume > 1.5x 20-day average
# Exit when: Price crosses back through the 20-day midpoint (mean of high and low)
# Target: 10-25 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "1d_Donchian20_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_max = high_max_20[i]
        low_min = low_min_20[i]
        midpoint = donchian_mid[i]
        ema50 = ema50_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-day high, weekly EMA50 upward, volume spike
            if (price > high_max and 
                ema50 > ema50_1w_aligned[i-1] and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-day low, weekly EMA50 downward, volume spike
            elif (price < low_min and 
                  ema50 < ema50_1w_aligned[i-1] and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below 20-day midpoint
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above 20-day midpoint
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals