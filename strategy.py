#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-week EMA trend filter and volume confirmation
# Long when price breaks above 20-period high, weekly EMA(21) uptrend, and volume spike
# Short when price breaks below 20-period low, weekly EMA(21) downtrend, and volume spike
# Weekly EMA filters for higher timeframe trend alignment
# Volume spike confirms institutional participation; avoids false breakouts
# Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

name = "4h_Donchian20_1wEMA21_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    ema21_1w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21_1w_val = ema21_1w_aligned[i]
        price = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above 20-period high, weekly uptrend, volume spike
            if price > high_20_val and price > ema21_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low, weekly downtrend, volume spike
            elif price < low_20_val and price < ema21_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below 20-period low or weekly trend turns down
            if price < low_20_val or price < ema21_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above 20-period high or weekly trend turns up
            if price > high_20_val or price > ema21_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals