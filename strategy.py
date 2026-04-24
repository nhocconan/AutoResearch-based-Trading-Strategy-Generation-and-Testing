#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and Camarilla levels.
- Camarilla pivot levels calculated from previous 4h OHLC.
- Entry: Long when price breaks above H3 with volume spike and close > 4h EMA50.
         Short when price breaks below L3 with volume spike and close < 4h EMA50.
- Exit: When price returns to the Camarilla R3/S3 levels (mean reversion edge).
- Uses session filter (08-20 UTC) to avoid low-liquidity hours.
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
    h3 = camarilla_close + range_val * 1.1 / 2
    l3 = camarilla_close - range_val * 1.1 / 2
    r3 = camarilla_close + range_val * 1.1 / 4
    s3 = camarilla_close - range_val * 1.1 / 4
    return r3, r3, r3, s3, s3, s3, h3, l3  # r1,r2,r1,s1,s2 not used

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels for each 4h bar (H3/L3 for breakout, R3/S3 for exit)
    h3_4h = np.full(len(df_4h), np.nan)
    l3_4h = np.full(len(df_4h), np.nan)
    r3_4h = np.full(len(df_4h), np.nan)
    s3_4h = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        _, _, _, _, _, _, h3, l3 = calculate_camarilla(
            df_4h['high'].iloc[i],
            df_4h['low'].iloc[i],
            df_4h['close'].iloc[i]
        )
        # Reuse function to get all levels, extract H3/L3 and R3/S3
        r3, _, _, _, _, s3, h3_val, l3_val = calculate_camarilla(
            df_4h['high'].iloc[i],
            df_4h['low'].iloc[i],
            df_4h['close'].iloc[i]
        )
        h3_4h[i] = h3_val
        l3_4h[i] = l3_val
        r3_4h[i] = r3
        s3_4h[i] = s3
    
    # Align 4h indicators to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 4h bars for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > H3 and close > EMA50
                if close[i] > h3_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakdown: price < L3 and close < EMA50
                elif close[i] < l3_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price returns to R3 (mean reversion)
            if close[i] <= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to S3 (mean reversion)
            if close[i] >= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0