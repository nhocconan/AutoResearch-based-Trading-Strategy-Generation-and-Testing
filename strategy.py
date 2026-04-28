#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend confirmation
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 12h data for breakout signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian(20) for breakout signals
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h volume confirmation (20-period average)
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: both 1d and 1w EMA must agree
        uptrend = close[i] > ema_34_1d_aligned[i] and close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i] and close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > (volume_ma_12h_aligned[i] * 1.5)
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and uptrend and vol_confirm
        short_entry = short_breakout and downtrend and vol_confirm
        
        # Exit conditions: reverse when trend changes
        if position == 1:
            # Long position: exit if trend turns down
            long_exit = not uptrend
        elif position == -1:
            # Short position: exit if trend turns up
            short_exit = not downtrend
        else:
            long_exit = False
            short_exit = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1d1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0