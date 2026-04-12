#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter (use previous week's value)
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_prev = np.roll(ema_200_1w, 1)  # previous week's value
    ema_200_1w_prev[0] = np.nan
    
    # Align weekly EMA200 to daily timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w_prev)
    
    # Calculate daily Camarilla pivot levels from previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's data (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    
    # Camarilla levels (H3 and L3 - mean reversion levels)
    h3_prev = pivot_prev + (range_1d_prev * 1.1 / 4)
    l3_prev = pivot_prev - (range_1d_prev * 1.1 / 4)
    
    # Align levels to daily timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    
    # Volume filter - 20-period average on daily data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine market regime based on weekly EMA200
        bull_regime = close[i] > ema_200_aligned[i]
        bear_regime = close[i] < ema_200_aligned[i]
        
        # Long: price crosses above H3 in bull regime with volume confirmation
        long_signal = (close[i] > h3_aligned[i] and 
                      close[i-1] <= h3_aligned[i-1] and  # crossed above
                      bull_regime and 
                      volume_ok[i])
        
        # Short: price crosses below L3 in bear regime with volume confirmation
        short_signal = (close[i] < l3_aligned[i] and 
                       close[i-1] >= l3_aligned[i-1] and  # crossed below
                       bear_regime and 
                       volume_ok[i])
        
        # Exit when price returns to pivot (mean reversion)
        pivot_prev_val = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev_val)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals