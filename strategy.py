#!/usr/bin/env python3

"""
Hypothesis: 12-hour Camarilla Pivot Level Reversal with 1-day Trend Filter and Volume Spike.
Camarilla levels provide natural support/resistance zones where price often reverses.
Using 1-day trend ensures we trade with the higher timeframe momentum, avoiding counter-trend traps.
Volume spikes confirm institutional interest at these key levels.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    # Camarilla levels
    close_val = close
    s1 = close - (range_val * 1.0833 / 2)
    s2 = close - (range_val * 1.1666 / 2)
    s3 = close - (range_val * 1.2500 / 2)
    s4 = close - (range_val * 1.5000 / 2)
    r1 = close + (range_val * 1.0833 / 2)
    r2 = close + (range_val * 1.1666 / 2)
    r3 = close + (range_val * 1.2500 / 2)
    r4 = close + (range_val * 1.5000 / 2)
    return s1, s2, s3, s4, r1, r2, r3, r4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    s1_1d, s2_1d, s3_1d, s4_1d, r1_1d, r2_1d, r3_1d, r4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate daily EMA for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Determine daily trend: price above/below EMA
    bullish_trend = close_1d > ema_34_1d
    bearish_trend = close_1d < ema_34_1d
    
    # Align Camarilla levels and trend to 12h timeframe
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume average (24-period ~ 12 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(s1_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches S3 level, bullish daily trend, volume spike
            if (low[i] <= s3_12h[i] * 1.001 and  # Allow small slippage
                bullish_aligned[i] > 0.5 and
                volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches R3 level, bearish daily trend, volume spike
            elif (high[i] >= r3_12h[i] * 0.999 and  # Allow small slippage
                  bearish_aligned[i] > 0.5 and
                  volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches S4 or shows weakness below S3
                if (low[i] <= s4_12h[i] * 1.001 or 
                    close[i] < s3_12h[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches R4 or shows strength above R3
                if (high[i] >= r4_12h[i] * 0.999 or 
                    close[i] > r3_12h[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_S3R3_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0