#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Squeeze_Breakout_v1
Hypothesis: Camarilla pivot breakouts on daily timeframe with weekly trend filter and volume confirmation capture major swing moves in both bull and bear markets. The squeeze condition (low volatility breakout) filters for high-probability explosive moves. Weekly EMA50 ensures alignment with higher timeframe trend. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Load daily data for Camarilla calculation (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Need at least 2 days for previous day calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on daily data (using previous day's OHLC)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R3/S3 are key breakout levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection on 1d (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    # Volatility squeeze condition: ATR(5) < ATR(20) * 0.7 (low volatility breakout)
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_5 = pd.Series(tr).ewm(span=5, adjust=False, min_periods=5).mean().values
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    volatility_squeeze = atr_5 < (atr_20 * 0.7)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(volatility_squeeze[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike + volatility squeeze + in uptrend
        if close[i] > r3_aligned[i] and volume_spike[i] and volatility_squeeze[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S3 with volume spike + volatility squeeze + in downtrend
        elif close[i] < s3_aligned[i] and volume_spike[i] and volatility_squeeze[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite level or trend weakens significantly
        elif position == 1 and (close[i] < s3_aligned[i] or close[i] < ema_50_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r3_aligned[i] or close[i] > ema_50_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Squeeze_Breakout_v1"
timeframe = "1d"
leverage = 1.0