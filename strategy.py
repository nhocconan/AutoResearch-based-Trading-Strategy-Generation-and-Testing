#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# Enter long when price breaks above 20-day high with volume > 1.5x average and 1w EMA > price
# Enter short when price breaks below 20-day low with volume > 1.5x average and 1w EMA < price
# Exit when price crosses 10-day EMA in opposite direction
# Uses 1d timeframe with 1w trend filter to capture medium-term trends while avoiding whipsaws
# Targets 50-100 total trades over 4 years (12-25/year) by requiring strong breakouts with volume and trend alignment

name = "1d_donchian_1w_ema_vol_v1"
timeframe = "1d"
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
    
    # 1d Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # 1w EMA for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below 10-day EMA
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above 10-day EMA
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and trend alignment
            # Long: break above 20-day high with volume and 1w EMA > price (uptrend)
            if (close[i] > highest_high[i] and volume[i] > volume_threshold[i] and 
                ema_20_1w_aligned[i] > close[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low with volume and 1w EMA < price (downtrend)
            elif (close[i] < lowest_low[i] and volume[i] > volume_threshold[i] and 
                  ema_20_1w_aligned[i] < close[i]):
                signals[i] = -0.25
                position = -1
    
    return signals