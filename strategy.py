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
    
    # Get weekly data for ATR-based volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get daily data for pivot points and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 12h timeframe
    daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Weekly volatility filter: current weekly ATR < 1.5 * 20-period average
    atr_ma20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_1w < (1.5 * atr_ma20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need weekly ATR20, daily EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_12h[i]) or 
            np.isnan(daily_r1_12h[i]) or 
            np.isnan(daily_s1_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Weekly volatility filter: only trade in low volatility
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_12h[i]
        price_below_s1 = close[i] < daily_s1_12h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with low volatility and above daily EMA50
            if (price_above_r1 and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with low volatility and below daily EMA50
            elif (price_below_s1 and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below daily EMA50
            if (close[i] < daily_pivot_12h[i]) or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above daily EMA50
            if (close[i] > daily_pivot_12h[i]) or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_EMA50_VolFilter"
timeframe = "12h"
leverage = 1.0