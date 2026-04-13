#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
    # Enter long when price breaks above R4 with volume > 1.5x 20-bar avg AND 1w close > 1w open (bullish week)
    # Enter short when price breaks below S4 with volume > 1.5x 20-bar avg AND 1w close < 1w open (bearish week)
    # Exit when price crosses the 1d midpoint (1d close)
    # Uses 1d HTF for Camarilla levels (more stable than 12h) and 1w for trend filter
    # Camarilla levels from 1d provide institutional support/resistance
    # Volume confirmation ensures breakouts have participation
    # Weekly trend filter avoids counter-trend trades in strong trends
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values  # for 1w trend calculation
    
    # Get 12h data for primary timeframe (used for alignment reference)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    cam_high_low = high_1d - low_1d
    camarilla_r4 = close_1d + (cam_high_low * 1.1 / 2)
    camarilla_s4 = close_1d - (cam_high_low * 1.1 / 2)
    camarilla_mid = close_1d  # midpoint is the 1d close
    
    # Calculate 1w trend: bullish if weekly close > weekly open
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Align 1w trend to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_r4_aligned[i]  # break above R4
        breakout_down = close[i] < camarilla_s4_aligned[i]  # break below S4
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = breakout_up and volume_confirmed[i] and weekly_bullish_aligned[i] > 0.5 and position != 1
        short_entry = breakout_down and volume_confirmed[i] and weekly_bearish_aligned[i] > 0.5 and position != -1
        
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

name = "12h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0