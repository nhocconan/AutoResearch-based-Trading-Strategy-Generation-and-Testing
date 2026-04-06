#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-period high AND weekly EMA(21) rising AND volume > 1.5x average
# Short when price breaks below 20-period low AND weekly EMA(21) falling AND volume > 1.5x average
# Exit when price crosses back through 10-period moving average
# Uses 6h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in both bull/bear markets by only trading with weekly trend

name = "6h_donchian20_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # 10-period EMA for exit
    ema_fast = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_fast[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back through 10-period EMA
        if position == 1:  # long position
            if close[i] <= ema_fast[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_fast[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with weekly trend confirmation
            # Long: price breaks above Donchian high AND weekly EMA rising AND volume confirmation
            if (close[i] > donchian_high[i] and 
                weekly_ema_aligned[i] > weekly_ema_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly EMA falling AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  weekly_ema_aligned[i] < weekly_ema_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals