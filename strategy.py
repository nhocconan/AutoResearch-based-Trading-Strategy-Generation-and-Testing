#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_12_1d = close_1d_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 26-period EMA on 1d
    ema_26_1d = close_1d_series.ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate MACD on 1d
    macd_line = ema_12_1d - ema_26_1d
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - macd_signal
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period SMA on 1w
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(250, n):
        # Skip if data not ready
        if (np.isnan(macd_hist_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # MACD histogram conditions
        macd_positive = macd_hist_aligned[i] > 0
        macd_negative = macd_hist_aligned[i] < 0
        
        # Weekly trend filter: price above/below weekly SMA20
        above_weekly_sma = close[i] > sma_20_1w_aligned[i]
        below_weekly_sma = close[i] < sma_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = macd_positive and above_weekly_sma
        short_entry = macd_negative and below_weekly_sma
        
        # Exit conditions: opposite MACD signal
        exit_long = position == 1 and macd_negative
        exit_short = position == -1 and macd_positive
        
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

name = "6h_1d_macd_hist_trend"
timeframe = "6h"
leverage = 1.0