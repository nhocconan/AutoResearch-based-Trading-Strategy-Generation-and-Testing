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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Align indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Donchian channels on daily
    donchian_period = 20
    upper_channel = np.maximum.accumulate(np.concatenate([np.full(donchian_period-1, np.nan), high_1d]))
    lower_channel = np.minimum.accumulate(np.concatenate([np.full(donchian_period-1, np.nan), low_1d]))
    
    # Align Donchian channels
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        volatility_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[max(0, i-100):i+1], 20)
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_aligned[i]
        breakout_down = close[i] < lower_aligned[i]
        
        # Entry conditions
        long_entry = uptrend and breakout_up and volatility_filter
        short_entry = downtrend and breakout_down and volatility_filter
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (breakout_down or not uptrend)
        exit_short = position == -1 and (breakout_up or not downtrend)
        
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

name = "1d_1w_donchian_breakout_ema50_trend_v1"
timeframe = "1d"
leverage = 1.0