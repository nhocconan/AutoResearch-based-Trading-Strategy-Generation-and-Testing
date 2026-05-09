# Your Name: [Your Name]
# Current date: 2025-07-22
# Additional context: 4h Camarilla pivot breakout with 1-day EMA trend filter and volume confirmation.
# This strategy aims to capture breakouts from key Camarilla pivot levels while filtering by higher timeframe trend and volume.
# It is designed to work in both bull and bear markets by using the 1-day EMA to determine trend direction and only taking trades in the direction of the trend.
# Volume confirmation helps avoid false breakouts.
# The Camarilla pivot levels are calculated from the previous day's high, low, and close.
# The strategy uses discrete position sizes (0.25) to minimize transaction costs.
# Target: 20-40 trades per year (80-160 total over 4 years).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for each day (using previous day's data)
    # We'll calculate these for each 1d bar and then align to 4h
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_r2 = np.full(len(df_1d), np.nan)
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    camarilla_s2 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's high, low, close to calculate today's Camarilla levels
        high_prev = df_1d['high'].iloc[i-1]
        low_prev = df_1d['low'].iloc[i-1]
        close_prev = df_1d['close'].iloc[i-1]
        
        # Camarilla pivot formulas
        range_prev = high_prev - low_prev
        camarilla_r4[i] = close_prev + range_prev * 1.1 / 2
        camarilla_r3[i] = close_prev + range_prev * 1.1 / 4
        camarilla_r2[i] = close_prev + range_prev * 1.1 / 6
        camarilla_r1[i] = close_prev + range_prev * 1.1 / 12
        camarilla_s1[i] = close_prev - range_prev * 1.1 / 12
        camarilla_s2[i] = close_prev - range_prev * 1.1 / 6
        camarilla_s3[i] = close_prev - range_prev * 1.1 / 4
        camarilla_s4[i] = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need 34 periods for EMA and 1 day for Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(camarilla_r2[i]) or np.isnan(camarilla_s2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla R1 AND price > 1d EMA34 (uptrend)
            if price > r1 and price > ema:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S1 AND price < 1d EMA34 (downtrend)
            elif price < s1 and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla S1 OR trend reverses (price < 1d EMA34)
            if price < s1 or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla R1 OR trend reverses (price > 1d EMA34)
            if price > r1 or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals