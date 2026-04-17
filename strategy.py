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
    open_time = pd.to_datetime(prices['open_time'])
    hours = open_time.dt.hour.values
    
    # Get 1d data for daily high/low and range calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's range (high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    daily_range = prev_high - prev_low
    
    # Define breakout levels: today's open +/- 0.3 * previous day's range (tighter than 0.5)
    # Today's open is previous day's close in crypto
    daily_open = np.roll(close_1d, 1)
    daily_open[0] = np.nan
    upper_break = daily_open + 0.3 * daily_range
    lower_break = daily_open - 0.3 * daily_range
    
    # Align daily breakout levels to 1h timeframe
    upper_break_1h = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_1h = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need volume MA20 and ATR MA20
    
    for i in range(start_idx, n):
        # Session filter: trade only between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(upper_break_1h[i]) or 
            np.isnan(lower_break_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average (stricter)
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        
        if position == 0:
            # Long: price breaks above upper level with volume and volatility
            if close[i] > upper_break_1h[i] and volume_filter and volatility_filter:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower level with volume and volatility
            elif close[i] < lower_break_1h[i] and volume_filter and volatility_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns below the breakout level or volatility drops
            if close[i] < upper_break_1h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns above the breakout level or volatility drops
            if close[i] > lower_break_1h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DailyRangeBreakout_VolVol_Tight_Session"
timeframe = "1h"
leverage = 1.0