#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_v1
Strategy: 12h Camarilla Pivot breakout with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12h price breakout above/below Camarilla Pivot levels (S4/S3 or R3/R4) confirmed by volume spike (>1.5x average volume) and filtered by 1d EMA50 trend direction. Captures strong momentum moves in trending markets while avoiding false breakouts in chop. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Camarilla Pivot levels (based on previous day's OHLC)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # S1 = Close - (Range * 1.1/2)
    # S2 = Close - (Range * 1.1)
    # S3 = Close - (Range * 1.1 * 2/2)
    # S4 = Close - (Range * 1.1 * 3/2)
    # R1 = Close + (Range * 1.1/2)
    # R2 = Close + (Range * 1.1)
    # R3 = Close + (Range * 1.1 * 2/2)
    # R4 = Close + (Range * 1.1 * 3/2)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    s1 = prev_close - (range_hl * 1.1 / 2)
    s2 = prev_close - (range_hl * 1.1)
    s3 = prev_close - (range_hl * 1.1 * 2)
    s4 = prev_close - (range_hl * 1.1 * 3)
    
    r1 = prev_close + (range_hl * 1.1 / 2)
    r2 = prev_close + (range_hl * 1.1)
    r3 = prev_close + (range_hl * 1.1 * 2)
    r4 = prev_close + (range_hl * 1.1 * 3)
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions: S3/S4 for long, R3/R4 for short
        breakout_up = price_close > s3_aligned[i]  # Above S3 = bullish breakout
        breakout_down = price_close < r3_aligned[i]  # Below R3 = bearish breakout
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout above S3 with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1d
        
        # Short: downward breakout below R3 with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when price returns to pivot
        exit_long = position == 1 and price_close < pivot_aligned[i]
        exit_short = position == -1 and price_close > pivot_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals