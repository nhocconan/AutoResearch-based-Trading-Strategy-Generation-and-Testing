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
    
    # Get 1d data for trend filter and Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    highest_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    highest_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    lowest_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    
    # Volume confirmation: current volume > 1.5x average volume (12h average)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_12h * 1.5
    
    # Choppiness index (14-period) on 1d timeframe
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close, 1))
    tr3 = np.abs(low_12h - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (14-period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chop
    chop = 100 * np.log10(hh_14 - ll_14) / np.log10(14) / np.log10(np.sum(tr, axis=0)) if False else np.zeros_like(high_12h)
    # Recalculate properly: sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(hh_14 - ll_14) / np.log10(tr_sum) / np.log10(14)
    chop = np.where(tr_sum > 0, chop, 50)  # Avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high_12h_aligned[i]) or
            np.isnan(lowest_low_12h_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_12h_aligned[i]
        breakout_down = close[i] < lowest_low_12h_aligned[i]
        
        # Choppiness filter: only trade in trending markets (CHOP < 38.2) or strong reversals in choppy (CHOP > 61.8)
        chop_value = chop_aligned[i]
        is_trending = chop_value < 38.2
        is_choppy = chop_value > 61.8
        
        # Entry conditions: require trend + breakout + volume confirmation + regime filter
        long_entry = uptrend and breakout_up and volume_confirm[i] and (is_trending or (is_choppy and chop_value > 61.8))
        short_entry = downtrend and breakout_down and volume_confirm[i] and (is_trending or (is_choppy and chop_value > 61.8))
        
        # Exit conditions: when trend reverses or opposite breakout or chop becomes extreme
        if position == 1:
            exit_condition = not uptrend or breakout_down or chop_value > 61.8
        elif position == -1:
            exit_condition = not downtrend or breakout_up or chop_value > 61.8
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "12h_Donchian20_1dEMA34_Volume_Chop"
timeframe = "12h"
leverage = 1.0