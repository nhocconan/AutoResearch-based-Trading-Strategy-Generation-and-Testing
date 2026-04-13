#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
    # Enter long when price breaks above R4 with volume > 2x avg and 4h close > 4h open
    # Enter short when price breaks below S4 with volume > 2x avg and 4h close < 4h open
    # Exit when price crosses the 1h close (midpoint)
    # Uses 4h HTF for trend filter (more reliable than 1h) and 1d for Camarilla levels
    # Session filter: 08-20 UTC to avoid low-volume Asian session
    # Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    cam_high_low = high_1d - low_1d
    camarilla_r4 = close_1d + (cam_high_low * 1.1 / 2)
    camarilla_s4 = close_1d - (cam_high_low * 1.1 / 2)
    camarilla_mid = close_1d  # midpoint is the 1d close
    
    # Align 1d Camarilla levels to 1h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Align 4h trend to 1h timeframe (trend = 1 if bullish, 0 if bearish)
    trend_4h = (close_4h > open_4h).astype(float)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):  # start from 20 to ensure volume MA is ready
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(avg_volume[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_r4_aligned[i]  # break above R4
        breakout_down = close[i] < camarilla_s4_aligned[i]  # break below S4
        
        # Entry conditions with volume confirmation and 4h trend filter
        long_entry = breakout_up and volume_confirmed[i] and (trend_4h_aligned[i] > 0.5) and position != 1
        short_entry = breakout_down and volume_confirmed[i] and (trend_4h_aligned[i] < 0.5) and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_mid_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_mid_aligned[i])
        
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

name = "1h_4h_1d_camarilla_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0