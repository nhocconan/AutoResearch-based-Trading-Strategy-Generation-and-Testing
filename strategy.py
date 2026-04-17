#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using previous day's data)
    # P = (H + L + C)/3
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    daily_r2 = daily_pivot + (high_1d - low_1d)
    daily_s2 = daily_pivot - (high_1d - low_1d)
    
    # Align daily pivot levels to 4h timeframe
    daily_pivot_4h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_4h = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_4h = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 120  # Need weekly EMA50, daily pivots, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_4h[i]) or 
            np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or 
            np.isnan(daily_r2_4h[i]) or 
            np.isnan(daily_s2_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema50_4h[i]
        price_below_weekly_ema = close[i] < ema50_4h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_4h[i]
        price_below_s1 = close[i] < daily_s1_4h[i]
        price_above_r2 = close[i] > daily_r2_4h[i]
        price_below_s2 = close[i] < daily_s2_4h[i]
        
        if position == 0:
            # Long: Price breaks above daily R2 with volume and above weekly EMA50
            if (price_above_r2 and price_above_weekly_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S2 with volume and below weekly EMA50
            elif (price_below_s2 and price_below_weekly_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily R1 OR below weekly EMA50
            if (close[i] < daily_r1_4h[i]) or (close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily S1 OR above weekly EMA50
            if (close[i] > daily_s1_4h[i]) or (close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_R2S2_WeeklyEMA50_Volume"
timeframe = "4h"
leverage = 1.0