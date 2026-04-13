#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
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
    vol_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR on 1d for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.zeros_like(tr)
    atr_14[13] = tr[1:14].mean()
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 50-period SMA on 1d (trend filter)
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period SMA on 1w
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(150, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below SMA50
        above_sma50 = close[i] > sma_50_1d_aligned[i]
        below_sma50 = close[i] < sma_50_1d_aligned[i]
        
        # Volatility filter: ATR > 0 (always true but keeps structure)
        vol_ok = atr_14_aligned[i] > 0
        
        # Weekly trend filter: price above/below weekly SMA20
        above_weekly_sma = close[i] > sma_20_1w_aligned[i]
        below_weekly_sma = close[i] < sma_20_1w_aligned[i]
        
        # Entry conditions: only trade in direction of both daily and weekly trend
        long_entry = above_sma50 and vol_ok and above_weekly_sma
        short_entry = below_sma50 and vol_ok and below_weekly_sma
        
        # Exit conditions: opposite trend signal
        exit_long = position == 1 and (below_sma50 or not above_weekly_sma)
        exit_short = position == -1 and (above_sma50 or not below_weekly_sma)
        
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

name = "1d_sma50_atr_weekly_trend"
timeframe = "1d"
leverage = 1.0