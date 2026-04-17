#!/usr/bin/env python3
"""
12h_1d_BullishTrend_Breakout_Volume
Strategy: Long-only breakout strategy for 12h timeframe with 1d trend filter.
- Long when price breaks above 1-day high with volume > 1.8x 20-period average and price above 1d EMA20
- Exit when price returns to 12h VWAP or breaks below 1-day low
- Designed to capture momentum in bull markets while avoiding bear market losses by being long-only
- Position size: 0.25
- Uses 12h timeframe as primary
"""

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
    
    # Calculate 12h VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get 1d data for breakout levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for trend filter (shorter for more sensitivity)
    close_series_1d = pd.Series(close_1d)
    ema20_1d = close_series_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d levels to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = max(20, 20)  # EMA20 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above 1d EMA20 (bullish bias)
        price_above_ema = close[i] > ema20_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_1d_aligned[i-1]  # break above previous day high
        breakout_down = close[i] < low_1d_aligned[i-1]  # break below previous day low
        
        # Return to 12h VWAP for exit
        return_to_vwap = abs(close[i] - vwap[i]) < 0.005 * close[i]  # within 0.5% of VWAP
        
        if position == 0:
            # Long: breakout up + volume filter + price above EMA (bullish only)
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: return to VWAP or break down (stop loss)
            if return_to_vwap or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "12h_1d_BullishTrend_Breakout_Volume"
timeframe = "12h"
leverage = 1.0