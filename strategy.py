#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation.
# Long when Bull Power > 0 (price > EMA13), Bear Power < 0 (price < EMA13), weekly trend up (price > weekly EMA26), and volume > 1.5x 20-period average.
# Short when Bull Power < 0, Bear Power > 0, weekly trend down (price < weekly EMA26), and volume confirmation.
# Uses Elder Ray to measure bull/bear power relative to EMA13, weekly EMA26 for trend filter, and volume for confirmation.
# Designed for 6h timeframe to capture swings in both bull and bear markets with proper trend alignment.
# Target: 50-150 total trades over 4 years (~12-37/year).
name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
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
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA26 for trend filter
    ema_26_weekly = pd.Series(close_weekly).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_26_weekly)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # Ensure weekly EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_26_weekly_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_13_val = ema_13[i]
        ema_26_weekly_val = ema_26_weekly_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if Bull Power > 0, Bear Power < 0, weekly trend up, and volume confirmation
            if bull > 0 and bear < 0 and price > ema_26_weekly_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if Bull Power < 0, Bear Power > 0, weekly trend down, and volume confirmation
            elif bull < 0 and bear > 0 and price < ema_26_weekly_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Bull Power <= 0 or Bear Power >= 0 (loss of bullish momentum)
            if bull <= 0 or bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Bull Power >= 0 or Bear Power <= 0 (loss of bearish momentum)
            if bull >= 0 or bear <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals