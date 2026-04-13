#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for weekly trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 34-period EMA on 1w (weekly trend)
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period Donchian on 1d (price channels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = rsi_14  # Already on daily
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume = df_1d['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    volume_ok = volume > (vol_ma20_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        below_weekly_ema = close[i] < ema_34_1w_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > high_20_aligned[i-1]  # Break above upper band
        donchian_breakout_down = close[i] < low_20_aligned[i-1]  # Break below lower band
        
        # RSI conditions: avoid extreme levels
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Volume confirmation
        vol_ok = volume_ok[i] if i < len(volume_ok) else False
        
        # Entry conditions: Donchian breakout + weekly trend + RSI + volume
        long_entry = (donchian_breakout_up and 
                     above_weekly_ema and 
                     rsi_not_overbought and 
                     vol_ok)
        
        short_entry = (donchian_breakout_down and 
                      below_weekly_ema and 
                      rsi_not_oversold and 
                      vol_ok)
        
        # Exit conditions: opposite Donchian breakout or RSI extreme
        exit_long = (position == 1 and 
                    (donchian_breakout_down or rsi_14_aligned[i] > 80))
        exit_short = (position == -1 and 
                     (donchian_breakout_up or rsi_14_aligned[i] < 20))
        
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

name = "1d_weekly_ema_donchian_volume_rsi_filter"
timeframe = "1d"
leverage = 1.0