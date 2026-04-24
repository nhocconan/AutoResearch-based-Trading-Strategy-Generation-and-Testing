#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above R3 AND 12h trend bullish AND volume spike.
         Short when price breaks below S3 AND 12h trend bearish AND volume spike.
- Exit: Opposite Camarilla level (S3 for long, R3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Camarilla levels provide precise intraday support/resistance that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get daily OHLC
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    # We focus on R3 and S3 for breakouts
    camarilla_r3 = daily_close + ((daily_high - daily_low) * 1.1 / 4)
    camarilla_s3 = daily_close - ((daily_high - daily_low) * 1.1 / 4)
    
    # Align daily Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period volume MA on 4h
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period volume MA
    volume_spike = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above R3 AND 12h trend bullish (price > EMA34)
                if curr_high > r3_val and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below S3 AND 12h trend bearish (price < EMA34)
                elif curr_low < s3_val and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR loss of volume confirmation
            if curr_low < s3_val or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR loss of volume confirmation
            if curr_high > r3_val or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0