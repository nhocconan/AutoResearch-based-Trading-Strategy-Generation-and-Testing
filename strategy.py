#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Index with weekly trend filter for trend following.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Weekly EMA34 determines trend: price > EMA34 = bullish, price < EMA34 = bearish.
# In bullish weekly trend: long when Bull Power > 0 and rising, exit when Bull Power <= 0.
# In bearish weekly trend: short when Bear Power < 0 and falling, exit when Bear Power >= 0.
# Volume confirmation: volume > 1.5x 20-period average to avoid chop.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "12h_ElderRay_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend determination
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align indicators to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Ensure EMA34 (34), EMA13 (13), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34_val = ema34_1w_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Determine weekly trend
        is_bullish_trend = price > ema34_val
        is_bearish_trend = price < ema34_val
        
        if position == 0:
            # Look for entries
            if is_bullish_trend and volume_confirmed:
                # Bullish trend: look for long when Bull Power is positive and rising
                if bull_power_val > 0 and bull_power_val > bull_power_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
            elif is_bearish_trend and volume_confirmed:
                # Bearish trend: look for short when Bear Power is negative and falling
                if bear_power_val < 0 and bear_power_val < bear_power_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or trend changes
            if bull_power_val <= 0 or not is_bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive or trend changes
            if bear_power_val >= 0 or not is_bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals