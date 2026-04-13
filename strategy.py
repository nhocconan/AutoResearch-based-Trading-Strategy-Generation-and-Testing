#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
    # Enter long when price breaks above H3 with 4h bullish trend (close > open) and volume > 1.5x avg
    # Enter short when price breaks below L3 with 4h bearish trend (close < open) and volume > 1.5x avg
    # Exit when price reaches H4/L4 levels or opposite Camarilla level
    # Uses 4h for signal direction (trend), 1h for entry timing precision
    # Session filter: 08-20 UTC to avoid low-volume Asian session noise
    # Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter (primary HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # 4h trend: bullish if close > open, bearish if close < open
    trend_bullish_4h = close_4h > open_4h
    trend_bearish_4h = close_4h < open_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Get 1d data for Camarilla pivot levels (secondary HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_h4_1d = close_1d + 1.1 * range_1d * 1.1 / 2.0
    camarilla_h3_1d = close_1d + 1.1 * range_1d * 1.1 / 4.0
    camarilla_l3_1d = close_1d - 1.1 * range_1d * 1.1 / 4.0
    camarilla_l4_1d = close_1d - 1.1 * range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(avg_volume[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > camarilla_h3_aligned[i-1]  # break above H3
        breakout_l3 = close[i] < camarilla_l3_aligned[i-1]  # break below L3
        
        # Entry conditions with 4h trend filter and volume confirmation
        long_entry = breakout_h3 and trend_bullish_aligned[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_l3 and trend_bearish_aligned[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions: reach H4/L4 or opposite Camarilla level
        exit_long = (position == 1 and (close[i] >= camarilla_h4_aligned[i] or close[i] <= camarilla_l3_aligned[i]))
        exit_short = (position == -1 and (close[i] <= camarilla_l4_aligned[i] or close[i] >= camarilla_h3_aligned[i]))
        
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

name = "1h_4h_1d_camarilla_breakout_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0