#!/usr/bin/env python3
"""
4h_12h_camarilla_pivot_volume_v1
Strategy: 4h Camarilla pivot breakout with volume confirmation and 12h trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 4h price breakout above/below daily Camarilla pivot levels (S3/S4 or R3/R4) confirmed by volume spike (>1.5x average volume) and filtered by 12h EMA50 trend direction. Camarilla levels provide precise support/resistance, reducing false breakouts. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
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
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day's data
    # We'll use 1d data to calculate Camarilla for the current day
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values   # Previous day high
    prev_low = df_1d['low'].shift(1).values     # Previous day low
    
    # Camarilla pivot calculation
    # Pivot = (prev_high + prev_low + prev_close) / 3
    # Range = prev_high - prev_low
    # S1 = close - (range * 1.1 / 12)
    # S2 = close - (range * 1.1 / 6)
    # S3 = close - (range * 1.1 / 4)
    # S4 = close - (range * 1.1 / 2)
    # R1 = close + (range * 1.1 / 12)
    # R2 = close + (range * 1.1 / 6)
    # R3 = close + (range * 1.1 / 4)
    # R4 = close + (range * 1.1 / 2)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    
    s1 = prev_close - (rng * 1.1 / 12)
    s2 = prev_close - (rng * 1.1 / 6)
    s3 = prev_close - (rng * 1.1 / 4)
    s4 = prev_close - (rng * 1.1 / 2)
    
    r1 = prev_close + (rng * 1.1 / 12)
    r2 = prev_close + (rng * 1.1 / 6)
    r3 = prev_close + (rng * 1.1 / 4)
    r4 = prev_close + (rng * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    
    # 4h EMA50 for trend filter (using 12h data for smoother trend)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(s4_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(r4_12h[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend_12h = price_close > ema_50_12h_aligned[i]
        downtrend_12h = price_close < ema_50_12h_aligned[i]
        
        # Breakout conditions: using S3/S4 for long, R3/R4 for short
        # S3/S4 represent strong support, R3/R4 represent strong resistance
        breakout_long = price_close > s3_12h[i] and price_close <= s4_12h[i]  # Price moved above S3 but below S4
        breakout_short = price_close < r3_12h[i] and price_close >= r4_12h[i]  # Price moved below R3 but above R4
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: price breaks above S3 (support) with volume in uptrend
        long_signal = breakout_long and vol_confirmed and uptrend_12h
        
        # Short: price breaks below R3 (resistance) with volume in downtrend
        short_signal = breakout_short and vol_confirmed and downtrend_12h
        
        # Exit when price returns to opposite level (mean reversion within the day)
        exit_long = position == 1 and price_close < s4_12h[i]  # Price returned below S4
        exit_short = position == -1 and price_close > r4_12h[i]  # Price returned above R4
        
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