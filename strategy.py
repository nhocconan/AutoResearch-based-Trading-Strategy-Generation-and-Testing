#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# Works in bull (breakouts with trend) and bear (fades false breaks via volume filter)
# Target: 12-37 trades/year, low frequency to avoid fee drag
# Uses 1d EMA for trend, 12h Donchian for entry, volume spike for confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_20 = np.full(len(high_12h), np.nan)
    low_20 = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        high_20[i] = np.max(high_12h[i-20:i])
        low_20[i] = np.min(low_12h[i-20:i])
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume spike filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry: breakout in trend direction with volume
        long_entry = long_breakout and above_ema and vol_confirm
        short_entry = short_breakout and below_ema and vol_confirm
        
        # Exit: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_ema)
        exit_short = position == -1 and (long_breakout or above_ema)
        
        # Execute
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_ema50_volume"
timeframe = "12h"
leverage = 1.0