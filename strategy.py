#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1-day Trend Filter and Volume Confirmation.
Goes long when price touches S1 and reverses up in an uptrend with volume spike.
Goes short when price touches R1 and reverses down in a downtrend with volume spike.
Exits when price reaches the opposite pivot level (R3 for longs, S3 for shorts) or trend reverses.
Designed for low frequency (12-37 trades/year) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), 
    # R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12), 
    # S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Using previous day's data for current day's levels (no look-ahead)
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate pivot levels for each day
    camarilla_R1 = np.empty_like(d_close, dtype=np.float64)
    camarilla_S1 = np.empty_like(d_close, dtype=np.float64)
    camarilla_R3 = np.empty_like(d_close, dtype=np.float64)
    camarilla_S3 = np.empty_like(d_close, dtype=np.float64)
    
    for i in range(len(d_close)):
        H = d_high[i]
        L = d_low[i]
        C = d_close[i]
        range_val = H - L
        camarilla_R1[i] = C + (range_val * 1.1 / 12)
        camarilla_S1[i] = C - (range_val * 1.1 / 12)
        camarilla_R3[i] = C + (range_val * 1.1 / 4)
        camarilla_S3[i] = C - (range_val * 1.1 / 4)
    
    # Align pivot levels to 12h timeframe (previous day's levels apply to current day)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.3x average (calculated from 12h volume MA20)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily data aligned
    start_idx = 19  # for volume MA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        r3 = camarilla_R3_aligned[i]
        s3 = camarilla_S3_aligned[i]
        daily_trend = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 1.3x average
        vol_filter = vol_now > 1.3 * vol_ma_20[i]
        
        # Reversal conditions: price near pivot level and reversing
        # For long: price touches S1 and starts moving up (close > open and close > S1)
        # For short: price touches R1 and starts moving down (close < open and close < R1)
        open_price = prices['open'].iloc[i]
        near_s1 = abs(price_now - s1) / s1 < 0.003  # within 0.3% of S1
        near_r1 = abs(price_now - r1) / r1 < 0.003  # within 0.3% of R1
        
        # Bullish reversal at S1: price at S1 and closing higher than open
        bullish_reversal = near_s1 and close[i] > open_price
        # Bearish reversal at R1: price at R1 and closing lower than open
        bearish_reversal = near_r1 and close[i] < open_price
        
        if position == 0:
            # Bull: reversal at S1 + daily trend up + volume
            if bullish_reversal and price_now > daily_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: reversal at R1 + daily trend down + volume
            elif bearish_reversal and price_now < daily_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R3 or daily trend turns down
            if price_now >= r3 or price_now < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S3 or daily trend turns up
            if price_now <= s3 or price_now > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1S1_Reversal_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0