#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h trend filter + volume confirmation
# Works in bull/bear: Breakouts capture momentum, 12h EMA filter avoids counter-trend trades, volume reduces false breakouts
# Targets: 12-37 trades/year (50-150 over 4 years) by requiring confluence of 3 filters
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 12h EMA50
        ema_50_i = align_htf_to_ltf(prices, df_12h, ema_50_12h)[i]
        
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Break above Donchian upper + 12h uptrend + volume
        if position == 0 and high[i] > highest_high[i] and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Break below Donchian lower + 12h downtrend + volume
        elif position == 0 and low[i] < lowest_low[i] and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite Donchian break or trend reversal
        elif position != 0:
            if position == 1 and low[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and high[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_12hEMA_Volume_Breakout"
timeframe = "6h"
leverage = 1.0