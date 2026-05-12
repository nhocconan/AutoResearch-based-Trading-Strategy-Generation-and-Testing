#!/usr/bin/env python3
name = "1h_Camarilla_Pivot_Volume_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (based on previous day's HLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # R2 = Close + (High - Low) * 1.166
    # R1 = Close + (High - Low) * 1.083
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.083
    # S2 = Close - (High - Low) * 1.166
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    range_hl = prev_high - prev_low
    r4 = prev_close + range_hl * 1.5
    r3 = prev_close + range_hl * 1.25
    r2 = prev_close + range_hl * 1.166
    r1 = prev_close + range_hl * 1.083
    pp = (prev_high + prev_low + prev_close) / 3
    s1 = prev_close - range_hl * 1.083
    s2 = prev_close - range_hl * 1.166
    s3 = prev_close - range_hl * 1.25
    s4 = prev_close - range_hl * 1.5
    
    # Align to 1h timeframe (use previous day's levels)
    r4_1h = align_htf_to_ltf(prices, df_1d, r4)
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    r2_1h = align_htf_to_ltf(prices, df_1d, r2)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    s2_1h = align_htf_to_ltf(prices, df_1d, s2)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    s4_1h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if Camarilla data not ready or outside session
        if np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation
            if (close[i] > r1_1h[i]) and vol_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 with volume confirmation
            elif (close[i] < s1_1h[i]) and vol_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price below S1 (mean reversion) or volume dries up
            if (close[i] < s1_1h[i]) or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price above R1 or volume dries up
            if (close[i] > r1_1h[i]) or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals