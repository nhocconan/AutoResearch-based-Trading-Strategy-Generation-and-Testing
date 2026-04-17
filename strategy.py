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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for price channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily channels to daily timeframe (no alignment needed as already daily)
    high_max_20_1d = high_max_20
    low_min_20_1d = low_min_20
    volume_ma20_1d = volume_ma20
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need 20-period indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_1d[i]) or 
            np.isnan(high_max_20_1d[i]) or 
            np.isnan(low_min_20_1d[i]) or 
            np.isnan(volume_ma20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema20_1d[i]
        price_below_weekly_ema = close[i] < ema20_1d[i]
        
        # Volume filter: current daily volume > 1.5 * 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20_1d[i])
        
        # Price relative to daily Donchian channels
        price_above_upper = close[i] > high_max_20_1d[i]
        price_below_lower = close[i] < low_min_20_1d[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with weekly uptrend and volume
            if (price_above_upper and price_above_weekly_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with weekly downtrend and volume
            elif (price_below_lower and price_below_weekly_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below lower Donchian OR weekly trend turns down
            if (close[i] < low_min_20_1d[i]) or (close[i] < ema20_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above upper Donchian OR weekly trend turns up
            if (close[i] > high_max_20_1d[i]) or (close[i] > ema20_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA20_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0