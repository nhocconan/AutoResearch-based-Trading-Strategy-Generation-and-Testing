#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1-week trend filter + volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum).
# Short when Bull Power < 0 and Bear Power > 0 (bearish momentum).
# Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50.
# Volume confirmation: volume > 1.5x 20-period average to avoid low-volume false signals.
# Designed for 6h timeframe to capture multi-day momentum while filtering counter-trend noise.
# Target: 50-150 total trades over 4 years = 12-37/year.
name = "6h_ElderRay_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure weekly EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if bullish momentum, above weekly EMA50, and volume confirmation
            if bull > 0 and bear < 0 and price > ema_50_1w_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if bearish momentum, below weekly EMA50, and volume confirmation
            elif bull < 0 and bear > 0 and price < ema_50_1w_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when momentum turns bearish or price crosses below weekly EMA50
            if bull < 0 or bear > 0 or price < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when momentum turns bullish or price crosses above weekly EMA50
            if bull > 0 or bear < 0 or price > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals