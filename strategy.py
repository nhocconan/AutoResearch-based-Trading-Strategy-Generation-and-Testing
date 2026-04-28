#!/usr/bin/env python3
"""
6h_Aroon_Breakaway_Trend_Strength
Hypothesis: Aroon(25) detects strong trending moves when Aroon-Up > 80 or Aroon-Down > 80, combined with volume confirmation and 1-week trend filter to avoid false breakouts in ranging markets. Designed for 6h timeframe to capture multi-day trends in BTC/ETH with controlled trade frequency (~15-25 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Aroon(25) calculation
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period, n):
        # Periods since highest high
        highest_high_idx = np.argmax(high[i-aroon_period:i+1])
        periods_since_high = aroon_period - highest_high_idx
        aroon_up[i] = ((aroon_period - periods_since_high) / aroon_period) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low[i-aroon_period:i+1])
        periods_since_low = aroon_period - lowest_low_idx
        aroon_down[i] = ((aroon_period - periods_since_low) / aroon_period) * 100
    
    # Volume confirmation: >1.6x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, aroon_period)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(aroon_up[i]) or
            np.isnan(aroon_down[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Aroon breakout conditions
        aroon_bullish = aroon_up[i] > 80 and aroon_down[i] < 30  # Strong uptrend
        aroon_bearish = aroon_down[i] > 80 and aroon_up[i] < 30  # Strong downtrend
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.6 * vol_ma_20[i])
        
        # Entry logic: Aroon breakout in direction of weekly trend with volume
        long_entry = vol_confirm and uptrend and aroon_bullish
        short_entry = vol_confirm and downtrend and aroon_bearish
        
        # Exit logic: Aroon weakening or trend change
        long_exit = aroon_up[i] < 50 or (not uptrend)
        short_exit = aroon_down[i] < 50 or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Aroon_Breakaway_Trend_Strength"
timeframe = "6h"
leverage = 1.0