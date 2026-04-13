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
    
    # Get weekly data for trend filter (HTF)
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
    
    # Align weekly indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily Donchian channels for entry signals
    donchian_period = 20
    upper_channel = np.zeros(n)
    lower_channel = np.zeros(n)
    
    for i in range(n):
        if i < donchian_period:
            upper_channel[i] = np.max(close[:i+1])
            lower_channel[i] = np.min(close[:i+1])
        else:
            upper_channel[i] = np.max(close[i-donchian_period+1:i+1])
            lower_channel[i] = np.min(close[i-donchian_period+1:i+1])
    
    # Calculate daily volume average for confirmation
    vol_ma = np.zeros(n)
    vol_period = 20
    for i in range(n):
        if i < vol_period:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-vol_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        volatility_threshold = atr_1w_aligned[i] * 1.5
        price_change = np.abs(close[i] - close[i-1])
        sufficient_volatility = price_change > volatility_threshold
        
        # Volume confirmation: above average volume
        volume_confirmation = volume[i] > vol_ma[i] * 1.2
        
        # Donchian breakout signals
        long_breakout = close[i] > upper_channel[i]
        short_breakout = close[i] < lower_channel[i]
        
        # Entry conditions
        long_entry = uptrend and long_breakout and sufficient_volatility and volume_confirmation
        short_entry = downtrend and short_breakout and sufficient_volatility and volume_confirmation
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or not uptrend)
        exit_short = position == -1 and (long_breakout or not downtrend)
        
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

name = "1d_1w_donchian_breakout_trend_vol_filter_v1"
timeframe = "1d"
leverage = 1.0