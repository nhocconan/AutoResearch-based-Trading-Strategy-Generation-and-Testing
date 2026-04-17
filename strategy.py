#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (classic)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot levels to 4h timeframe
    weekly_pivot_4h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_4h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_4h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_4h[i]) or 
            np.isnan(weekly_r1_4h[i]) or 
            np.isnan(weekly_s1_4h[i]) or 
            np.isnan(ema34_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_4h[i]
        price_below_ema = close[i] < ema34_4h[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_4h[i]
        price_below_s1 = close[i] < weekly_s1_4h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above weekly EMA34
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below weekly EMA34
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below weekly EMA34
            if (close[i] < weekly_pivot_4h[i]) or (close[i] < ema34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above weekly EMA34
            if (close[i] > weekly_pivot_4h[i]) or (close[i] > ema34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyPivot_Breakout_EMA34_Volume"
timeframe = "4h"
leverage = 1.0