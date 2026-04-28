#!/usr/bin/env python3
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
    
    # Get daily data for pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # P = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # R4 = C + (H - L) * 1.1 / 2
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 / 4
    # S4 = C - (H - L) * 1.1 / 2
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    r3 = pivot + (range_val * 1.1 / 4.0)
    s3 = pivot - (range_val * 1.1 / 4.0)
    r4 = pivot + (range_val * 1.1 / 2.0)
    s4 = pivot - (range_val * 1.1 / 2.0)
    
    # Calculate EMA34 on daily for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x average of last 4 periods (24h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema34_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema34_6h[i]
        downtrend = close[i] < ema34_6h[i]
        
        # Fade at S3/R3 (mean reversion)
        long_fade = (close[i] <= s3_6h[i]) and uptrend and vol_filter[i]
        short_fade = (close[i] >= r3_6h[i]) and downtrend and vol_filter[i]
        
        # Breakout continuation at S4/R4 (trend following)
        long_breakout = (close[i] >= s4_6h[i]) and uptrend and vol_filter[i]
        short_breakout = (close[i] <= r4_6h[i]) and downtrend and vol_filter[i]
        
        # Exit conditions
        long_exit = (position == 1) and (close[i] >= pivot[i] or not uptrend)
        short_exit = (position == -1) and (close[i] <= pivot[i] or not downtrend)
        
        # Entry logic
        if (long_fade or long_breakout) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_fade or short_breakout) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_S3R3_Fade_S4R4_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0