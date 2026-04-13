#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend + volume confirmation
    # Long: price > Donchian(20) high AND 12h EMA(50) rising AND volume > 1.5x avg
    # Short: price < Donchian(20) low AND 12h EMA(50) falling AND volume > 1.5x avg
    # Exit: opposite Donchian break or EMA direction change
    # Uses 4h timeframe for balance of trade frequency and signal quality
    # Discrete position sizing (0.25) to minimize fee churn
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h EMA direction (1 = rising, -1 = falling, 0 = flat)
    ema_dir = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(ema_12h_aligned[i]) and not np.isnan(ema_12h_aligned[i-1]):
            if ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                ema_dir[i] = 1
            elif ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                ema_dir[i] = -1
            else:
                ema_dir[i] = ema_dir[i-1]
        else:
            ema_dir[i] = 0
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_dir[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: Donchian breakout + EMA trend + volume confirmation
        long_entry = (close[i] > donchian_high[i]) and (ema_dir[i] == 1) and volume_spike[i]
        short_entry = (close[i] < donchian_low[i]) and (ema_dir[i] == -1) and volume_spike[i]
        
        # Exit logic: opposite Donchian break or EMA direction change
        long_exit = (close[i] < donchian_low[i]) or (ema_dir[i] == -1)
        short_exit = (close[i] > donchian_high[i]) or (ema_dir[i] == 1)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0