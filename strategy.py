#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high with weekly EMA50 uptrend and volume > 1.5x average
# Short when price breaks below 20-day low with weekly EMA50 downtrend and volume > 1.5x average
# Exit when price retouches the 10-day moving average (mean reversion in ranges)
# Uses weekly trend filter to avoid counter-trend trades, volume for conviction, Donchian for breakouts
# Designed for low-frequency, high-conviction trades in both trending and ranging markets
# Target: 40-80 total trades over 4 years (10-20/year) with size 0.25

name = "1d_Donchian_20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 10-day moving average for exit (mean reversion target)
    ma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-period Donchian channels (breakout levels)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ma10[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high, weekly EMA50 uptrend, volume confirmation
            if (close[i] > donchian_high[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # Weekly EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, weekly EMA50 downtrend, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # Weekly EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches 10-day mean (mean reversion in ranges)
            if close[i] <= ma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches 10-day mean (mean reversion in ranges)
            if close[i] >= ma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals