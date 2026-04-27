#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot reversal with 1-day volume confirmation and 1-day trend filter.
Trades reversals at Camarilla S1/R1 levels when volume exceeds 1-day average and intraday trend confirms.
Designed to work in both bull and bear markets by using daily trend as filter and volume to confirm reversal strength.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drift.
"""

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
    
    # Get 4-hour data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4-hour Camarilla levels (using previous day's OHLC)
    # We'll use the 4-hour bar's high/low/close as proxy for intraday calculation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous day's range
    # For intraday, we use the 4-bar lookback to approximate previous session
    range_4h = pd.Series(high_4h).rolling(window=4, min_periods=4).max() - \
               pd.Series(low_4h).rolling(window=4, min_periods=4).min()
    close_prev = pd.Series(close_4h).shift(4)  # approximate previous day's close
    
    # Avoid lookback issues by using shifted values
    range_val = range_4h.values
    close_val = close_prev.values
    
    # Camarilla levels
    S1 = close_val + (range_val * 1.0 / 12)
    S2 = close_val + (range_val * 2.0 / 12)
    S3 = close_val + (range_val * 3.0 / 12)
    S4 = close_val + (range_val * 4.0 / 12)
    R1 = close_val + (range_val * 11.0 / 12)
    R2 = close_val + (range_val * 12.0 / 12)
    R3 = close_val + (range_val * 13.0 / 12)
    R4 = close_val + (range_val * 14.0 / 12)
    
    # Align Camarilla levels to 4-hour timeframe
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    S4_aligned = align_htf_to_ltf(prices, df_4h, S4)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    R4_aligned = align_htf_to_ltf(prices, df_4h, R4)
    
    # Get daily data for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(25) for trend
    close_1d = df_1d['close'].values
    ema_25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla levels, volume MA, and daily EMA
    start_idx = max(4, 20, 25)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_25_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 4-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_25_1d_aligned[i]
        
        # Current Camarilla levels
        S1_now = S1_aligned[i]
        R1_now = R1_aligned[i]
        
        # Volume filter: volume > 1.3x 1-day average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Camarilla reversal with volume and daily trend alignment
        if position == 0:
            # Long: price at S1 with volume + daily uptrend
            if price_now <= S1_now and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price at R1 with volume + daily downtrend
            elif price_now >= R1_now and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S2 or daily trend turns down
            S2_now = S2_aligned[i]
            if price_now >= S2_now or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R2 or daily trend turns up
            R2_now = R2_aligned[i]
            if price_now <= R2_now or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_CamarillaS1R1_Reversal_1dVolume_1dTrend"
timeframe = "4h"
leverage = 1.0