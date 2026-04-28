#!/usr/bin/env python3
"""
4h_GoldenRatio_Breakout_Volume_Trend
Hypothesis: Golden ratio (0.618) extensions from prior swing highs/lows act as institutional support/resistance. 
Breakouts above 0.618 extension of prior swing low (long) or below 0.618 extension of prior swing high (short) 
with volume confirmation and trend filter (EMA50) capture strong moves. Designed for 4h timeframe to work in 
both bull (breakouts continue) and bear (breakouts fail/reverse) markets. Target: 20-40 trades/year to minimize 
fee drag while capturing significant moves.
"""

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
    
    # Get 1d data for trend filter and swing points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate swing highs/lows on 1d (using 3-bar lookback)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Swing high: current high > previous high and next high
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(1, len(high_1d)-1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high[i] = True
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low[i] = True
    
    # Get most recent swing high/low values
    last_swing_high = np.full(len(high_1d), np.nan)
    last_swing_low = np.full(len(low_1d), np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(len(high_1d)):
        if swing_high[i]:
            last_high_val = high_1d[i]
        if swing_low[i]:
            last_low_val = low_1d[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # Calculate 0.618 extensions
    # For longs: extend up from swing low
    # For shorts: extend down from swing high
    range_from_low = high_1d - last_swing_low
    ext_618_above_low = last_swing_low + range_from_low * 0.618
    
    range_from_high = last_swing_high - low_1d
    ext_618_below_high = last_swing_high - range_from_high * 0.618
    
    # Align to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ext_618_above_low_aligned = align_htf_to_ltf(prices, df_1d, ext_618_above_low)
    ext_618_below_high_aligned = align_htf_to_ltf(prices, df_1d, ext_618_below_high)
    
    # Volume spike: current volume > 1.8x 24-period average (4h * 6 = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ext_618_above_low_aligned[i]) or 
            np.isnan(ext_618_below_high_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > ext_618_above_low_aligned[i-1]
        breakout_short = close[i] < ext_618_below_high_aligned[i-1]
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_GoldenRatio_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0