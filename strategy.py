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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 7d data for volatility regime filter
    df_7d = get_htf_data(prices, '7d')
    if len(df_7d) < 14:
        return np.zeros(n)
    
    close_7d = df_7d['close'].values
    high_7d = df_7d['high'].values
    low_7d = df_7d['low'].values
    
    # 7d ATR(14) for volatility regime
    tr1 = high_7d[1:] - low_7d[1:]
    tr2 = np.abs(high_7d[1:] - close_7d[:-1])
    tr3 = np.abs(low_7d[1:] - close_7d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_7d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_7d_aligned = align_htf_to_ltf(prices, df_7d, atr_7d, additional_delay_bars=0)
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Volume confirmation: current volume > 1.5x average volume (4h average)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_4h * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high_4h_aligned[i]) or
            np.isnan(lowest_low_4h_aligned[i]) or
            np.isnan(atr_7d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when volatility is elevated
        # Use current ATR vs 7-period average of ATR to detect expansion
        if i >= 7:
            atr_avg = np.nanmean(atr_7d_aligned[i-6:i+1])
            vol_expansion = atr_7d_aligned[i] > atr_avg * 1.2
        else:
            vol_expansion = True  # default to true for early bars
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_4h_aligned[i]
        breakout_down = close[i] < lowest_low_4h_aligned[i]
        
        # Entry conditions: require trend + breakout + volume confirmation + vol expansion
        long_entry = uptrend and breakout_up and volume_confirm[i] and vol_expansion
        short_entry = downtrend and breakout_down and volume_confirm[i] and vol_expansion
        
        # Exit conditions: when trend reverses or opposite breakout
        if position == 1:
            exit_condition = not uptrend or breakout_down
        elif position == -1:
            exit_condition = not downtrend or breakout_up
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA34_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0