#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action with 12h EMA trend filter and volume confirmation
# Long when price > 12h EMA50 + volume > 1.5x 20-period average + price makes higher high
# Short when price < 12h EMA50 + volume > 1.5x 20-period average + price makes lower low
# Exit when price crosses back below/above EMA
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_EMA50_Trend_HigherHighLowerLow_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average volume
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        vol_filter = volume[i] > 1.5 * vol_ma_20
        
        if position == 0:
            # Look for entry: EMA trend + volume + higher high/lower low
            higher_high = high[i] > high[i-1]
            lower_low = low[i] < low[i-1]
            
            long_condition = close[i] > ema_50_aligned[i] and vol_filter and higher_high
            short_condition = close[i] < ema_50_aligned[i] and vol_filter and lower_low
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA
            if close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals