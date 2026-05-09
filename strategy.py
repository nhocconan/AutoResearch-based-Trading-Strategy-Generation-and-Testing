#!/usr/bin/env python3
# Hypothesis: 4h price breaking above/below 1-day high/low with volume spike and 1-week trend filter
# Long when price > 1-day high, volume > 2x 20-period average, and price > 1-week EMA50
# Short when price < 1-day low, volume > 2x 20-period average, and price < 1-week EMA50
# Exit when price returns inside 1-day range OR weekly trend contradicts position
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed to capture breakouts in trending markets while avoiding false breakouts in ranging markets

name = "4h_Breakout_1dRange_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align 1d high/low to 4h timeframe (waits for daily close)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above daily high + volume spike + above weekly EMA50 (bullish trend)
            if (close[i] > daily_high_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema50_1w[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below daily low + volume spike + below weekly EMA50 (bearish trend)
            elif (close[i] < daily_low_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema50_1w[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below daily high OR weekly trend turns bearish
            if (close[i] < daily_high_aligned[i]) or (close[i] < ema50_1w[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above daily low OR weekly trend turns bullish
            if (close[i] > daily_low_aligned[i]) or (close[i] > ema50_1w[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals