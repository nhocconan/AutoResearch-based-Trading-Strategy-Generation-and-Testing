#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d context filters to reduce noise
# Strategy: 4h EMA trend + 1d Donchian breakout + volume confirmation
# Entry: 4h EMA21 trend aligned + price breaks 1d Donchian(20) + volume > 1.5x MA
# Exit: Opposite Donchian break or trend reversal
# Time filters: 08-20 UTC to avoid low-liquidity hours
# Position size: 0.20 (20%) to control drawdown
# Target: 15-30 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h EMA(21) for trend
    close_4h_series = pd.Series(close_4h)
    ema_21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d Donchian channel (20)
    donch_high_20 = np.full(len(df_1d), np.nan)
    donch_low_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donch_high_20[i] = np.max(high_1d[i-19:i+1])
        donch_low_20[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 1h volume MA(20)
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute hour filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price > 4h EMA21 for long, < for short
        trend_long = close[i] > ema_21_4h_aligned[i]
        trend_short = close[i] < ema_21_4h_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > donch_high_20_aligned[i]
        breakout_short = close[i] < donch_low_20_aligned[i]
        
        # Entry conditions: trend + breakout + volume
        long_entry = trend_long and breakout_long and vol_filter
        short_entry = trend_short and breakout_short and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = close[i] < donch_low_20_aligned[i] or not trend_long
        short_exit = close[i] > donch_high_20_aligned[i] or not trend_short
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_ema_trend_breakout_vol"
timeframe = "1h"
leverage = 1.0