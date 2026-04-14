#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses daily EMA200 to filter counter-trend trades in bear/bull markets
# Donchian(20) provides clear entry/exit with low churn
# Volume spike confirms breakout strength
# Target: 12-37 trades/year (50-150 over 4 years) with strict entry conditions
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1d EMA200
        ema_200_i = align_htf_to_ltf(prices, df_1d, ema_200_1d)[i]
        
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_200_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Price breaks above Donchian high + above 1d EMA200 + volume
        if position == 0 and high[i] > highest_high[i-1] and close[i] > ema_200_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below Donchian low + below 1d EMA200 + volume
        elif position == 0 and low[i] < lowest_low[i-1] and close[i] < ema_200_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite Donchian breakout
        elif position != 0:
            if position == 1 and low[i] < lowest_low[i-1]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and high[i] > highest_high[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0