#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-day average.
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-day average.
# Exit when price returns to 10-day moving average (mean reversion) or opposite breakout occurs.
# Uses 1d price action with weekly trend filter to avoid counter-trend trades in both bull and bear markets.
# Target: 30-80 total trades over 4 years (7-20/year) for low fee drift.

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 10-day and 20-day moving averages for exit
    ma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 20-day Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above 20-day high, above 1w EMA50, volume spike
            long_cond = (close[i] > highest_high[i]) and (close[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: break below 20-day low, below 1w EMA50, volume spike
            short_cond = (close[i] < lowest_low[i]) and (close[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: return to 10-day MA or opposite breakout
            if close[i] < ma10[i] or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: return to 10-day MA or opposite breakout
            if close[i] > ma10[i] or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals