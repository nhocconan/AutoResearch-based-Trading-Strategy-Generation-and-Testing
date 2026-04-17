#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and ATR
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14) for volatility filter and position sizing
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align weekly EMA200 and ATR to 12h timeframe
    ema200_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    atr14_12h = align_htf_to_ltf(prices, df_1w, atr14)
    
    # Align daily pivot levels to 12h timeframe
    daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Volume filter: current volume > 2.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need weekly EMA200, ATR14, daily pivot, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_12h[i]) or 
            np.isnan(atr14_12h[i]) or 
            np.isnan(daily_pivot_12h[i]) or 
            np.isnan(daily_r1_12h[i]) or 
            np.isnan(daily_s1_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0 (avoid dead markets)
        vol_filter = atr14_12h[i] > 0
        
        # Volume filter
        volume_filter = volume[i] > (2.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA200
        price_above_ema200 = close[i] > ema200_12h[i]
        price_below_ema200 = close[i] < ema200_12h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_12h[i]
        price_below_s1 = close[i] < daily_s1_12h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume, volatility, and above weekly EMA200
            if (price_above_r1 and price_above_ema200 and vol_filter and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume, volatility, and below weekly EMA200
            elif (price_below_s1 and price_below_ema200 and vol_filter and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below weekly EMA200
            if (close[i] < daily_pivot_12h[i]) or (close[i] < ema200_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above weekly EMA200
            if (close[i] > daily_pivot_12h[i]) or (close[i] > ema200_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA200_DailyPivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0