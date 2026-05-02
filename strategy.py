#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot (R3/S3) breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivots identify key intraday support/resistance levels. Breakouts above R3 or below S3
# with volume confirmation and 4h trend alignment capture strong momentum moves.
# Works in bull markets (buy R3 breakouts in uptrend) and bear markets (sell S3 breakdowns in downtrend).
# Uses 4h/1d for signal direction, 1h only for entry timing to minimize trade frequency.
# Session filter (08-20 UTC) reduces noise trades. Target: 15-37 trades/year on 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot levels (more stable than 4h pivots)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    df_1d['h1'] = df_1d['high'].shift(1)
    df_1d['l1'] = df_1d['low'].shift(1)
    df_1d['c1'] = df_1d['close'].shift(1)
    
    # Calculate pivot and levels
    pivot = (df_1d['h1'] + df_1d['l1'] + df_1d['c1']) / 3
    range_hl = df_1d['h1'] - df_1d['l1']
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    
    # Align to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation (2.0x 20-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume)
    start_idx = 55  # max(50 for EMA, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > R3 + 4h uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: close < S3 + 4h downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: close < R3 or trend reversal
            if close[i] < r3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: close > S3 or trend reversal
            if close[i] > s3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals