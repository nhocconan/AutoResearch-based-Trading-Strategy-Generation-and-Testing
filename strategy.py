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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly
    high_20w = np.full(len(close_1w), np.nan)
    low_20w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        high_20w[i] = np.max(high_1w[i-20:i])
        low_20w[i] = np.min(low_1w[i-20:i])
    
    # Calculate 50-period EMA on weekly
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR(14) for volatility filter and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or 
            np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Weekly Donchian breakout conditions
        long_breakout = close[i] > high_20w_aligned[i]
        short_breakout = close[i] < low_20w_aligned[i]
        
        # Entry conditions: breakout in direction of trend
        long_entry = long_breakout and above_ema
        short_entry = short_breakout and below_ema
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_ema)
        exit_short = position == -1 and (long_breakout or above_ema)
        
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

name = "1d_1w_donchian_ema50_breakout"
timeframe = "1d"
leverage = 1.0