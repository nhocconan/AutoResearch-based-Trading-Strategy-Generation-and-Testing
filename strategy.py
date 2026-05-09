#!/usr/bin/env python3
# Hypothesis: 1d Donchian channel breakout with 1w EMA trend filter and volume spike
# Long when price breaks above 20-day high with 1w EMA uptrend and volume > 2x average
# Short when price breaks below 20-day low with 1w EMA downtrend and volume > 2x average
# Exit when price returns to 10-day EMA (mean reversion within trend)
# Uses Donchian for breakout, EMA for trend filter, volume for conviction
# Target: 40-80 total trades over 4 years (10-20/year) with size 0.25

name = "1d_Donchian_Breakout_1wEMA_Trend_Volume"
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
    
    # Calculate 1d Donchian channels (20-period)
    high_roll = pd.Series(high)
    low_roll = pd.Series(low)
    donch_high = high_roll.rolling(window=20, min_periods=20).max().values
    donch_low = low_roll.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d 10-period EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1w EMA34 for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema10[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 1w EMA uptrend, volume spike
            if (close[i] > donch_high[i] and 
                ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 1w EMA downtrend, volume spike
            elif (close[i] < donch_low[i] and 
                  ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 10-day EMA (mean reversion within trend)
            if close[i] <= ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 10-day EMA (mean reversion within trend)
            if close[i] >= ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals