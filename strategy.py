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
    
    # Get 1d data for ATR-based channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.zeros_like(tr1))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period) on 1d
    high_max20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    atr14_6h = align_htf_to_ltf(prices, df_1d, atr14)
    high_max20_6h = align_htf_to_ltf(prices, df_1d, high_max20)
    low_min20_6h = align_htf_to_ltf(prices, df_1d, low_min20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need ATR, Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr14_6h[i]) or 
            np.isnan(high_max20_6h[i]) or 
            np.isnan(low_min20_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema50_6h[i]
        price_below_weekly_ema = close[i] < ema50_6h[i]
        
        if position == 0:
            # Long: Break above 20-day high with volume and uptrend
            if (close[i] > high_max20_6h[i] and price_above_weekly_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-day low with volume and downtrend
            elif (close[i] < low_min20_6h[i] and price_below_weekly_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below 20-day low OR trend reverses
            if (close[i] < low_min20_6h[i]) or (close[i] < ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above 20-day high OR trend reverses
            if (close[i] > high_max20_6h[i]) or (close[i] > ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0