#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA20 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and Camarilla levels.
- Camarilla pivot levels calculated from previous 4h OHLC.
- Entry: Long when price breaks above H3 with volume spike and close > 4h EMA20.
         Short when price breaks below L3 with volume spike and close < 4h EMA20.
- Exit: When price returns to the Camarilla R3/S3 levels (mean reversion edge).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    # Get 4h data for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20 = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels for each 4h bar
    r3_4h = np.full(len(df_4h), np.nan)
    r2_4h = np.full(len(df_4h), np.nan)
    r1_4h = np.full(len(df_4h), np.nan)
    s1_4h = np.full(len(df_4h), np.nan)
    s2_4h = np.full(len(df_4h), np.nan)
    s3_4h = np.full(len(df_4h), np.nan)
    h3_4h = np.full(len(df_4h), np.nan)
    l3_4h = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        r3, r2, r1, s1, s2, s3, h3, l3 = calculate_camarilla(
            df_4h['high'].iloc[i],
            df_4h['low'].iloc[i],
            df_4h['close'].iloc[i]
        )
        r3_4h[i] = r3
        r2_4h[i] = r2
        r1_4h[i] = r1
        s1_4h[i] = s1
        s2_4h[i] = s2
        s3_4h[i] = s3
        h3_4h[i] = h3
        l3_4h[i] = l3
    
    # Align 4h indicators to 1h
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA (on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need enough 4h bars for EMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > H3 and close > EMA20
                if close[i] > h3_aligned[i] and close[i] > ema_20_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakdown: price < L3 and close < EMA20
                elif close[i] < l3_aligned[i] and close[i] < ema_20_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price returns to R3 (mean reversion) or stoploss
            if close[i] <= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to S3 (mean reversion) or stoploss
            if close[i] >= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0