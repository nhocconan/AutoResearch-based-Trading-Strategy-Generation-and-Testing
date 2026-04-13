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
    
    # Get 1d data for weekly context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily Donchian(20) breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper (20-day high)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-day low)
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate daily ATR(14) for stop loss
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1]) if len(close_1d) > 1 else np.array([])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1]) if len(close_1d) > 1 else np.array([])
    close_1d = df_1d['close'].values
    tr_first = np.array([np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])])
    if len(tr2) > 0 and len(tr3) > 0:
        tr = np.concatenate([tr_first, np.maximum(tr1, np.maximum(tr2, tr3))])
    else:
        tr = tr_first
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above weekly EMA21 = uptrend, below = downtrend
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions: Donchian breakout with trend filter
        long_entry = uptrend and close[i] > donch_high_aligned[i]
        short_entry = downtrend and close[i] < donch_low_aligned[i]
        
        # Exit conditions: opposite Donchian breakout or ATR stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: price breaks below Donchian low or hits ATR stop
            if close[i] < donch_low_aligned[i]:
                exit_long = True
            elif i > 0:  # ATR stop
                entry_price_approx = close[i-1]  # approximate entry from previous bar
                if close[i] < entry_price_approx - 2.0 * atr_aligned[i]:
                    exit_long = True
        elif position == -1:
            # Exit short: price breaks above Donchian high or hits ATR stop
            if close[i] > donch_high_aligned[i]:
                exit_short = True
            elif i > 0:  # ATR stop
                entry_price_approx = close[i-1]  # approximate entry from previous bar
                if close[i] > entry_price_approx + 2.0 * atr_aligned[i]:
                    exit_short = True
        
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

name = "1d_20_donchian_weekly_ema_trend"
timeframe = "1d"
leverage = 1.0