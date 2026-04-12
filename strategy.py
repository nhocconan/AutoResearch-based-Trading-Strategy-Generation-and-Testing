#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with weekly trend filter
    # Uses weekly EMA200 for trend filter: only take breakouts aligned with weekly trend
    # Entry: Camarilla R4 breakout long in weekly uptrend, S4 breakdown short in weekly downtrend
    # Exit: Opposite Camarilla level (R3/S3) or trend reversal
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC (bar i-1 in 1d data corresponds to prior 24h period)
        prev_idx = min(i-1, len(df_1d)-1)
        if prev_idx >= 0:
            ph = df_1d['high'].iloc[prev_idx]
            pl = df_1d['low'].iloc[prev_idx]
            pc = df_1d['close'].iloc[prev_idx]
            rang = ph - pl
            if rang > 0:
                camarilla_r4[i] = pc + rang * 1.1 / 2
                camarilla_r3[i] = pc + rang * 1.1 / 4
                camarilla_s3[i] = pc - rang * 1.1 / 4
                camarilla_s4[i] = pc - rang * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_s4[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        # Entry logic: Camarilla breakout with weekly trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Camarilla R4 in weekly uptrend
        if bullish_trend:
            long_entry = (close[i] > camarilla_r4[i])
        # Short breakout: price breaks below Camarilla S4 in weekly downtrend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_s4[i])
        
        # Exit logic: Opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_s3[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_r3[i]) or (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_camarilla_breakout_trend_filter_v1"
timeframe = "6h"
leverage = 1.0