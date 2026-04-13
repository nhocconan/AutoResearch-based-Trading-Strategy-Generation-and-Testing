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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 10-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_10_1d = close_1d_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period SMA on 1w
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_10_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA10
        above_ema = close[i] > ema_10_aligned[i]
        below_ema = close[i] < ema_10_aligned[i]
        
        # RSI conditions: avoid extreme levels
        rsi_not_overbought = rsi_14_aligned[i] < 80
        rsi_not_oversold = rsi_14_aligned[i] > 20
        
        # Weekly trend filter: price above/below weekly SMA20
        above_weekly_sma = close[i] > sma_20_1w_aligned[i]
        below_weekly_sma = close[i] < sma_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = above_ema and rsi_not_overbought and above_weekly_sma
        short_entry = below_ema and rsi_not_oversold and below_weekly_sma
        
        # Exit conditions: opposite signal or RSI extreme
        exit_long = position == 1 and (below_ema or rsi_14_aligned[i] > 85)
        exit_short = position == -1 and (above_ema or rsi_14_aligned[i] < 15)
        
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

name = "12h_ema10_rsi14_weekly_sma20_filter"
timeframe = "12h"
leverage = 1.0