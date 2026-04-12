#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_reversion
Uses weekly Camarilla pivot levels on 1d timeframe for mean reversion.
Enters long when price touches weekly S3/S4 with RSI < 30 and volume confirmation.
Enters short when price touches weekly R3/R4 with RSI > 70 and volume confirmation.
Exits when price returns to weekly pivot (PP) or reverses.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
Works in ranging markets and catches reversals in trending markets.
"""

name = "1d_1w_camarilla_pivot_reversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    pp = (high + low + close) / 3
    r4 = close + range_val * 1.500
    r3 = close + range_val * 1.250
    r2 = close + range_val * 1.166
    r1 = close + range_val * 1.083
    s1 = close - range_val * 1.083
    s2 = close - range_val * 1.166
    s3 = close - range_val * 1.250
    s4 = close - range_val * 1.500
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def calculate_rsi(series, period=14):
    """Calculate RSI with proper handling"""
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(series)
    avg_loss = np.zeros_like(series)
    
    # Wilder's smoothing
    if len(series) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    pp_1w = np.full_like(close_1w, np.nan)
    r3_1w = np.full_like(close_1w, np.nan)
    r4_1w = np.full_like(close_1w, np.nan)
    s3_1w = np.full_like(close_1w, np.nan)
    s4_1w = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            high_1w[i], low_1w[i], close_1w[i]
        )
        pp_1w[i] = pp
        r3_1w[i] = r3
        r4_1w[i] = r4
        s3_1w[i] = s3
        s4_1w[i] = s4
    
    # Align weekly Camarilla levels to daily timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily RSI for overbought/oversold confirmation
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(r4_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches S3/S4 with RSI < 30 and volume confirmation
        if ((close[i] <= s3_1w_aligned[i] * 1.002 or close[i] <= s4_1w_aligned[i] * 1.002) and
            rsi[i] < 30 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price touches R3/R4 with RSI > 70 and volume confirmation
        elif ((close[i] >= r3_1w_aligned[i] * 0.998 or close[i] >= r4_1w_aligned[i] * 0.998) and
              rsi[i] > 70 and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to pivot or reverses
        elif position == 1 and (close[i] >= pp_1w_aligned[i] * 0.998 or 
                                close[i] <= s3_1w_aligned[i] * 0.995):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= pp_1w_aligned[i] * 1.002 or 
                                 close[i] >= r3_1w_aligned[i] * 1.005):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals