#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for pivot points and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 6h timeframe
    daily_pivot_6h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_6h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_6h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily EMA20 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_6h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need weekly EMA50, daily EMA20, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(daily_pivot_6h[i]) or 
            np.isnan(daily_r1_6h[i]) or 
            np.isnan(daily_s1_6h[i]) or 
            np.isnan(ema20_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Weekly trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        # Daily trend filter: price above/below daily EMA20
        price_above_daily_ema = close[i] > ema20_6h[i]
        price_below_daily_ema = close[i] < ema20_6h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_6h[i]
        price_below_s1 = close[i] < daily_s1_6h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume, above both weekly and daily EMA
            if (price_above_r1 and price_above_weekly_ema and price_above_daily_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume, below both weekly and daily EMA
            elif (price_below_s1 and price_below_weekly_ema and price_below_daily_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below daily EMA20
            if (close[i] < daily_pivot_6h[i]) or (close[i] < ema20_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above daily EMA20
            if (close[i] > daily_pivot_6h[i]) or (close[i] > ema20_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyEMA50_DailyPivot_R1S1_Volume"
timeframe = "6h"
leverage = 1.0