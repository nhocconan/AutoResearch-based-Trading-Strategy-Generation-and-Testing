#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Camarilla pivot levels.
- Camarilla pivot levels calculated from previous 1d OHLC.
- Entry: Long when price breaks above H3 with volume spike and close > 1d EMA34 (uptrend).
         Short when price breaks below L3 with volume spike and close < 1d EMA34 (downtrend).
- Exit: When price returns to the Camarilla R3/S3 levels (mean reversion edge).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    camarilla_close = close
    r3 = camarilla_close + range_val * 1.1 / 4
    r2 = camarilla_close + range_val * 1.1 / 6
    r1 = camarilla_close + range_val * 1.1 / 12
    s1 = camarilla_close - range_val * 1.1 / 12
    s2 = camarilla_close - range_val * 1.1 / 6
    s3 = camarilla_close - range_val * 1.1 / 4
    h3 = camarilla_close + range_val * 1.1 / 2
    l3 = camarilla_close - range_val * 1.1 / 2
    return r3, r2, r1, s1, s2, s3, h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each 1d bar
    r3_1d = np.full(len(df_1d), np.nan)
    r2_1d = np.full(len(df_1d), np.nan)
    r1_1d = np.full(len(df_1d), np.nan)
    s1_1d = np.full(len(df_1d), np.nan)
    s2_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    h3_1d = np.full(len(df_1d), np.nan)
    l3_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r3, r2, r1, s1, s2, s3, h3, l3 = calculate_camarilla(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
        r3_1d[i] = r3
        r2_1d[i] = r2
        r1_1d[i] = r1
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        h3_1d[i] = h3
        l3_1d[i] = l3
    
    # Align 1d indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > H3 and close > EMA34
                if close[i] > h3_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < L3 and close < EMA34
                elif close[i] < l3_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to R3 (mean reversion) or stoploss
            if close[i] <= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to S3 (mean reversion) or stoploss
            if close[i] >= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0