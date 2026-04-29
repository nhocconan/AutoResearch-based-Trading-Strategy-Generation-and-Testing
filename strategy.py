#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation + ATR stoploss
# Long when price breaks above 20-day Donchian high AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-day Donchian low AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit via ATR-based trailing stop: signal=0 when long and price < highest_high - 2.5*ATR, or short and price > lowest_low + 2.5*ATR
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 15-35 trades/year on 1d timeframe.
# Donchian channels provide robust trend-following structure, 1w EMA50 filters counter-trend moves on weekly timeframe,
# volume confirmation ensures breakout validity. ATR stoploss manages risk without look-ahead.
# Designed to work in both bull and bear markets via trend filter and volatility-based exits.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d data for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 1d data
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w data
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # track highest high since long entry
    lowest_low_since_entry = 0.0    # track lowest low since short entry
    
    start_idx = max(20, 14, 50)  # Donchian, ATR, and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_atr = atr_1d[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_ema50 = ema_50_1w_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle position exits and trailing stops
        if position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_high_since_entry:
                highest_high_since_entry = curr_high
            # ATR trailing stop: exit if price drops below highest_high - 2.5*ATR
            if curr_close < (highest_high_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0  # reset
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_low_since_entry:
                lowest_low_since_entry = curr_low
            # ATR trailing stop: exit if price rises above lowest_low + 2.5*ATR
            if curr_close > (lowest_low_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0  # reset
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above 20-day Donchian high AND price > 1w EMA50 AND volume confirmation
            if curr_close > curr_highest_20 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high  # initialize tracking
            # Short when price breaks below 20-day Donchian low AND price < 1w EMA50 AND volume confirmation
            elif curr_close < curr_lowest_20 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low  # initialize tracking
            else:
                signals[i] = 0.0
    
    return signals