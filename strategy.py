#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla breakout with 4h trend filter and session
    # Long: price breaks above H3 (4h) + 4h close > open (bullish) + UTC 08-20
    # Short: price breaks below L3 (4h) + 4h close < open (bearish) + UTC 08-20
    # Exit: price retreats to 4h midpoint (P) or opposite Camarilla level
    # Uses 4h Camarilla levels for structure, 1h for precise entry timing
    # Session filter reduces noise and fee drag
    # Target: 60-150 total trades over 4 years (15-37/year)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Calculate 4h Camarilla levels (based on previous day's range)
    # P = (high + low + close) / 3
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    # Use previous 4h bar's high/low/close for today's levels
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    # First bar will have NaN due to roll, handled by min_periods logic
    
    camarilla_p = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_h3 = camarilla_p + camarilla_range * 1.1 / 4
    camarilla_l3 = camarilla_p - camarilla_range * 1.1 / 4
    camarilla_h4 = camarilla_p + camarilla_range * 1.1 / 2
    camarilla_l4 = camarilla_p - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_4h, camarilla_p)
    
    # 4h trend filter: bullish if close > open, bearish if close < open
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h)
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h)
    
    # Session filter: UTC 08-20
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(1, n):  # start from 1 to have previous 4h bar
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Trend filter conditions
        long_trend = bullish_4h_aligned[i]
        short_trend = bearish_4h_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and long_trend and position != 1
        short_entry = short_breakout and short_trend and position != -1
        
        # Exit conditions: retreat to midpoint or opposite level
        exit_long = position == 1 and (close[i] < camarilla_p_aligned[i] or close[i] < camarilla_l3_aligned[i])
        exit_short = position == -1 and (close[i] > camarilla_p_aligned[i] or close[i] > camarilla_h3_aligned[i])
        
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

name = "1h_4h_camarilla_breakout_trend_session_v1"
timeframe = "1h"
leverage = 1.0