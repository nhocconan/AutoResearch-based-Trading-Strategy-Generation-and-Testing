#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(
        high_1w - low_1w,
        np.maximum(
            np.abs(high_1w - np.roll(close_1w, 1)),
            np.abs(low_1w - np.roll(close_1w, 1))
        )
    )
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = np.zeros_like(tr_1w)
    for i in range(len(tr_1w)):
        if i < 14:
            atr_1w[i] = np.mean(tr_1w[:i+1]) if i > 0 else tr_1w[i]
        else:
            atr_1w[i] = 0.93 * atr_1w[i-1] + 0.07 * tr_1w[i]
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility days
        daily_range = high[i] - low[i]
        volatility_filter = daily_range < (atr_1w_aligned[i] * 3.0)
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: Donchian breakout with trend alignment
        long_entry = (close[i] > high_max_20[i]) and uptrend and volatility_filter
        short_entry = (close[i] < low_min_20[i]) and downtrend and volatility_filter
        
        # Exit conditions: opposite Donchian breakout
        exit_long = position == 1 and (close[i] < low_min_20[i])
        exit_short = position == -1 and (close[i] > high_max_20[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_trend_filter_v1"
timeframe = "1d"
leverage = 1.0