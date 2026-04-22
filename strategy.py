#!/usr/bin/env python3

"""
12h Camarilla Pivot (R1/S1) breakout with 1d trend filter and volume confirmation.
Breakouts above R1 (long) or below S1 (short) on 12h chart, filtered by daily trend
(price above/below EMA34 on 1d). Volume spike confirms institutional interest.
Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_bullish = close_1d > ema_34_1d
    daily_bearish = close_1d < ema_34_1d
    
    # Align daily EMA trend to 12h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # Calculate 12h Camarilla levels using prior bar's OHLC
    # Shift by 1 to avoid look-ahead (use previous bar's data)
    pivot, r1, s1 = calculate_camarilla(high[:-1], low[:-1], close[:-1])
    # Prepend first value to maintain array length
    pivot = np.concatenate([[pivot[0]], pivot])
    r1 = np.concatenate([[r1[0]], r1])
    s1 = np.concatenate([[s1[0]], s1])
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, daily bullish, volume spike
            if (close[i] > r1[i] and 
                daily_bullish_aligned[i] > 0.5 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, daily bearish, volume spike
            elif (close[i] < s1[i] and 
                  daily_bearish_aligned[i] > 0.5 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot level
            exit_signal = False
            if position == 1:
                # Exit long: price drops back to pivot or below
                if close[i] <= pivot[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises back to pivot or above
                if close[i] >= pivot[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0