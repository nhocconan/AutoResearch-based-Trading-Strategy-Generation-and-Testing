#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h trend direction and 1d volume confirmation.
# Long when: 4h EMA50 > EMA100 (uptrend), 1d volume > 1.5x 20-period average (high conviction),
# and 1h price crosses above 1h EMA20 (pullback entry in uptrend).
# Short when: 4h EMA50 < EMA100 (downtrend), 1d volume > 1.5x 20-period average,
# and 1h price crosses below 1h EMA20 (pullback entry in downtrend).
# Exit when price crosses back over EMA20.
# Uses 1h timeframe as specified, with 4h trend and 1d volume for higher timeframe context.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency to avoid fee drag.

name = "1h_EMA20_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:
        return np.zeros(n)
    
    # 1d data for volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1h EMA20
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).values
    
    # 4h EMA50 and EMA100 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).values
    ema100_4h = pd.Series(close_4h).ewm(span=100, adjust=False, min_periods=100).values
    
    # Trend: EMA50 > EMA100 for uptrend, EMA50 < EMA100 for downtrend
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema100_4h)
    trend_up = ema50_4h_aligned > ema100_4h_aligned
    trend_down = ema50_4h_aligned < ema100_4h_aligned
    
    # 1d volume filter: current volume > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma20_1d)
    volume_filter = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 100, 20)  # EMA20, EMA50, EMA100, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema100_4h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 4h uptrend, high volume, price crosses above EMA20
            long_cond = trend_up[i] and volume_filter[i] and (close[i] > ema20[i]) and (close[i-1] <= ema20[i-1])
            # Short conditions: 4h downtrend, high volume, price crosses below EMA20
            short_cond = trend_down[i] and volume_filter[i] and (close[i] < ema20[i]) and (close[i-1] >= ema20[i-1])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA20
            if close[i] < ema20[i] and close[i-1] >= ema20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above EMA20
            if close[i] > ema20[i] and close[i-1] <= ema20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals