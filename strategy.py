#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long when: price breaks above H3 (resistance) AND 1w close > 1w open (bullish week) AND volume > 1.5x 20-period average
    # Short when: price breaks below L3 (support) AND 1w close < 1w open (bearish week) AND volume > 1.5x 20-period average
    # Exit when: price returns to mean (Pivot level) OR adverse 1w trend reversal
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # Works in bull/bear via 1w trend filter preventing counter-trend trades.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day's OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), H2 = close + 0.5*(high-low)
    # L3 = close - 1.0*(high-low), L2 = close - 0.5*(high-low), L1 = close - 1.5*(high-low)
    # Pivot = (high + low + close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day will have invalid data (rolled from last day) - will be filtered by min_periods logic
    
    # Calculate Camarilla levels for previous day
    camarilla_H3 = prev_close_1d + 1.0 * (prev_high_1d - prev_low_1d)  # H3 resistance
    camarilla_L3 = prev_close_1d - 1.0 * (prev_high_1d - prev_low_1d)  # L3 support
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0  # Pivot point
    
    # Align 1d levels to 1d timeframe (no additional delay needed as these are based on completed day)
    camarilla_H3_1d = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_1d = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_pivot_1d = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # 1w bullish/bearish trend (based on weekly close vs open)
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align 1w trend to 1d timeframe
    weekly_bullish_1d = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_1d = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_H3_1d[i]) or np.isnan(camarilla_L3_1d[i]) or 
            np.isnan(camarilla_pivot_1d[i]) or np.isnan(weekly_bullish_1d[i]) or 
            np.isnan(weekly_bearish_1d[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_H3_1d[i]
        short_breakout = close[i] < camarilla_L3_1d[i]
        
        # Entry conditions
        long_entry = (long_breakout and 
                     weekly_bullish_1d[i] > 0.5 and  # Bullish week
                     volume_confirmed[i] and 
                     position != 1)
                     
        short_entry = (short_breakout and 
                      weekly_bearish_1d[i] > 0.5 and  # Bearish week
                      volume_confirmed[i] and 
                      position != -1)
        
        # Exit conditions: return to pivot or adverse weekly trend
        exit_long = (position == 1 and 
                    (close[i] < camarilla_pivot_1d[i] or  # Price returned to pivot
                     weekly_bearish_1d[i] > 0.5))  # Weekly trend turned bearish
        
        exit_short = (position == -1 and 
                     (close[i] > camarilla_pivot_1d[i] or  # Price returned to pivot
                      weekly_bullish_1d[i] > 0.5))  # Weekly trend turned bullish
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0