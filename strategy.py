#!/usr/bin/env python3
"""
1d_1w_RangeBreakout_Volume_Trend
Strategy: Daily breakout of weekly high/low with volume confirmation and weekly trend filter.
Long: Price breaks above 1-week high + volume > 1.8x 20-day avg + price above weekly EMA20
Short: Price breaks below 1-week low + volume > 1.8x 20-day avg + price below weekly EMA20
Exit: Price returns to daily VWAP
Position size: 0.25
Designed to capture breakouts aligned with weekly trend in both bull and bear markets.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly levels to daily timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation (20-day MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need weekly EMA20 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1w_aligned[i]) or 
            np.isnan(low_1w_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-day average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA20
        price_above_ema = close[i] > ema20_1w_aligned[i]
        price_below_ema = close[i] < ema20_1w_aligned[i]
        
        # Breakout conditions (using previous day's weekly levels)
        breakout_up = close[i] > high_1w_aligned[i-1]  # break above previous week high
        breakout_down = close[i] < low_1w_aligned[i-1]  # break below previous week low
        
        # Return to daily VWAP for exit
        return_to_vwap = abs(close[i] - vwap[i]) < 0.005 * close[i]  # within 0.5% of VWAP
        
        if position == 0:
            # Long: breakout up + volume filter + price above weekly EMA
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + price below weekly EMA
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to VWAP or break down
            if return_to_vwap or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to VWAP or break up
            if return_to_vwap or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_RangeBreakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0