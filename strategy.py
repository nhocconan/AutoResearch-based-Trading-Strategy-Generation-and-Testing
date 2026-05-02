#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses 1w Camarilla pivot levels for long-term bias and 6h Donchian breakout for entry timing
# Entry: Long when price breaks above Donchian(20) high AND price > weekly R3 (bullish bias) AND volume spike
#        Short when price breaks below Donchian(20) low AND price < weekly S3 (bearish bias) AND volume spike
# Exit: Close crosses opposite Donchian(10) level (counter-trend breakout) OR weekly pivot bias flips
# Works in both bull and bear markets by aligning with weekly structure while capturing 6h momentum
# Target: 75-150 total trades over 4 years (19-38/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla pivot levels for trend bias (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # We need to shift by 1 week to avoid look-ahead: use previous week's data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    range_w = weekly_high - weekly_low
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = pivot + range_w * 1.1 / 2.0
    r4 = pivot + range_w * 1.1
    s3 = pivot - range_w * 1.1 / 2.0
    s4 = pivot - range_w * 1.1
    
    # Align weekly levels to 6h timeframe (with 1-week delay to avoid look-ahead)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3, additional_delay_bars=1)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3, additional_delay_bars=1)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4, additional_delay_bars=1)
    
    # Calculate 6h Donchian channels
    donchian_len = 20
    donchian_len_exit = 10  # shorter for exit
    
    # Upper band: highest high over period
    highest_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Exit bands (shorter period)
    highest_high_exit = pd.Series(high).rolling(window=donchian_len_exit, min_periods=donchian_len_exit).max().values
    lowest_low_exit = pd.Series(low).rolling(window=donchian_len_exit, min_periods=donchian_len_exit).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(100, donchian_len)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian breakout above AND price > weekly R3 (bullish bias) AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > r3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakdown below AND price < weekly S3 (bearish bias) AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Donchian(10) low OR weekly bias turns bearish (price < S3)
            if close[i] < lowest_low_exit[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Donchian(10) high OR weekly bias turns bullish (price > R3)
            if close[i] > highest_high_exit[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals