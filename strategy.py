#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot point reversal with 1d trend filter and volume confirmation
# Uses 1d Camarilla levels (S1, S2, R1, R2) for reversal zones, filtered by 1d EMA50 trend.
# Volume > 1.5x 20-period average confirms institutional interest.
# Designed for 12h timeframe with tight entries targeting 20-50 trades per year.
# Works in bull/bear by requiring trend alignment for entries.
name = "12h_Camarilla_R1S1_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+CLOSE)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_low_range = df_1d['high'] - df_1d['low']
    
    camarilla_r1 = typical_price + (high_low_range * 1.1 / 12)
    camarilla_s1 = typical_price - (high_low_range * 1.1 / 12)
    camarilla_r2 = typical_price + (high_low_range * 1.1 / 6)
    camarilla_s2 = typical_price - (high_low_range * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    r2_12h = align_htf_to_ltf(prices, df_1d, camarilla_r2.values)
    s2_12h = align_htf_to_ltf(prices, df_1d, camarilla_s2.values)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Camarilla calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Reversal conditions at Camarilla levels
        # Long setup: price touches S1/S2 and reverses up in uptrend
        long_setup = (close[i] <= s1_12h[i] * 1.002 or close[i] <= s2_12h[i] * 1.002) and close[i-1] > s1_12h[i-1]
        # Short setup: price touches R1/R2 and reverses down in downtrend
        short_setup = (close[i] >= r1_12h[i] * 0.998 or close[i] >= r2_12h[i] * 0.998) and close[i-1] < r1_12h[i-1]
        
        trend_up = close[i] > ema_50_12h[i]
        trend_down = close[i] < ema_50_12h[i]
        
        if position == 0:
            # Long: reversal up from support + uptrend + volume confirmation
            if long_setup and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: reversal down from resistance + downtrend + volume confirmation
            elif short_setup and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: reversal down from resistance or trend reversal
            short_exit = (close[i] >= r1_12h[i] * 0.998 or close[i] >= r2_12h[i] * 0.998) and close[i-1] < r1_12h[i-1]
            if short_exit or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: reversal up from support or trend reversal
            long_exit = (close[i] <= s1_12h[i] * 1.002 or close[i] <= s2_12h[i] * 1.002) and close[i-1] > s1_12h[i-1]
            if long_exit or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals