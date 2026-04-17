#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1_S1_Breakout_Volume_Trend_v1
1-hour strategy using 4-hour and 1-day timeframe alignment for direction and 1-hour for entry timing.
Enters long when price breaks above 4h R1 with volume above average, price above 4h EMA34, and 1d close > 1d open (bullish daily candle).
Enters short when price breaks below 4h S1 with volume above average, price below 4h EMA34, and 1d close < 1d open (bearish daily candle).
Uses session filter (08-20 UTC) and tight entry conditions to limit trades and avoid fee drag.
Target: 15-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4-hour Data (HTF for direction and structure) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4-hour Camarilla Pivot Levels (R1, S1)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align 4h levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 4-hour EMA34 for Trend Filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 4-hour Volume for Confirmation (20-period average)
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # === 1-day Data (HTF for bias filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Daily bullish/bearish bias (close > open = bullish, close < open = bearish)
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Align daily bias to 1h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period (ensure enough data for indicators)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(daily_bullish_aligned[i]) or
            np.isnan(daily_bearish_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h bar's volume for confirmation
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_confirmed = vol_4h_current > 1.5 * vol_ma_4h_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1_4h_aligned[i]
        breakout_short = close[i] < s1_4h_aligned[i]
        
        # Exit conditions: return to opposite pivot level
        exit_long = close[i] < s1_4h_aligned[i]
        exit_short = close[i] > r1_4h_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume confirmation, 4h trend filter, and daily bullish bias
            if breakout_long and vol_confirmed and close[i] > ema34_4h_aligned[i] and daily_bullish_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
                continue
            # Short: break below S1 with volume confirmation, 4h trend filter, and daily bearish bias
            elif breakout_short and vol_confirmed and close[i] < ema34_4h_aligned[i] and daily_bearish_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below S1
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R1
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_Volume_Trend_v1"
timeframe = "1h"
leverage = 1.0