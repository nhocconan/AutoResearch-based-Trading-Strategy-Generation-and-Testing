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
    
    # Get 12h data for calculations (primary HTF for 4h strategy)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-period ATR on 12h for volatility filter
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], 
                     np.maximum(np.abs(high_12h[1:] - close_12h[:-1]),
                                np.abs(low_12h[1:] - close_12h[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    atr1_12h = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    atr1_aligned = align_htf_to_ltf(prices, df_12h, atr1_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(atr1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr1_aligned[i] > 0
        
        # Trend filter: price above/below daily EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and vol_filter and above_ema
        short_entry = breakout_down and vol_filter and below_ema
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (breakout_down or atr1_aligned[i] <= 0)
        exit_short = position == -1 and (breakout_up or atr1_aligned[i] <= 0)
        
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

name = "4h_12h_donchian_breakout_ema50_vol_filter"
timeframe = "4h"
leverage = 1.0