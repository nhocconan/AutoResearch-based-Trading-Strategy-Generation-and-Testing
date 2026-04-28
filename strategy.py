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
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(20) - using weekly high/low of last 20 weeks
    # Calculate rolling max/min on weekly data
    highest_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h
    highest_20w_aligned = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(highest_20w_aligned[i]) or
            np.isnan(lowest_20w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_1d_aligned[i] > 0.005 * close[i]  # At least 0.5% ATR
        
        # Entry conditions: Weekly Donchian breakout with trend and volatility
        long_breakout = close[i] > highest_20w_aligned[i]
        short_breakout = close[i] < lowest_20w_aligned[i]
        
        long_entry = uptrend and long_breakout and vol_filter
        short_entry = downtrend and short_breakout and vol_filter
        
        # Exit conditions: Weekly Donchian opposite breakout or trend reversal
        long_exit = (not uptrend) or (close[i] < lowest_20w_aligned[i])
        short_exit = (not downtrend) or (close[i] > highest_20w_aligned[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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

name = "6h_WeeklyDonchian20_1dTrend_VolatilityFilter"
timeframe = "6h"
leverage = 1.0