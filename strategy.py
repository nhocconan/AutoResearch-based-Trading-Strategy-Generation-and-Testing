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
    
    # Get daily data for 1-day chart
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly pivot points from daily data (using 5-day window)
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    # Weekly resistance and support levels
    weekly_r1 = 2 * weekly_pivot - low_5d
    weekly_s1 = 2 * weekly_pivot - high_5d
    
    # Align weekly pivot levels to 1d timeframe
    weekly_pivot_1d = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_1d = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_1d = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current daily volume > 1.8 * 30-day average
    volume_ma30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    volume_ma30_aligned = align_htf_to_ltf(prices, df_1d, volume_ma30)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 80  # Need weekly pivot, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_1d[i]) or 
            np.isnan(weekly_r1_1d[i]) or 
            np.isnan(weekly_s1_1d[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma30_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.8 * 30-day average
        volume_filter = volume_1d[i] > (1.8 * volume_ma30_aligned[i])
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close_1d[i] > ema34_1d_aligned[i]
        price_below_ema = close_1d[i] < ema34_1d_aligned[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close_1d[i] > weekly_r1_1d[i]
        price_below_s1 = close_1d[i] < weekly_s1_1d[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above daily EMA34
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly S1 with volume and below daily EMA34
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below daily EMA34
            if (close_1d[i] < weekly_pivot_1d[i]) or (close_1d[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above daily EMA34
            if (close_1d[i] > weekly_pivot_1d[i]) or (close_1d[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_WeeklyPivot_Breakout_EMA34_Volume"
timeframe = "1d"
leverage = 1.0