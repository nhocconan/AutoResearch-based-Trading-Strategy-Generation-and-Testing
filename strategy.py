#!/usr/bin/env python3
"""
1h_4H_1D_Camarilla_Structure_V1
Hypothesis: Use 4h trend via EMA50 and 1d structure via Camarilla R3/S3 for directional bias,
with 1h for entry timing via breakout of prior 4h swing high/low during active session (08-20 UTC).
Long when 4h EMA50 > 4h EMA200 (bullish), price > 1d R3, and breaks above prior 4h swing high.
Short when 4h EMA50 < 4h EMA200 (bearish), price < 1d S3, and breaks below prior 4h swing low.
Volume > 1.5x 20-bar average confirms breakout. Position size 0.20.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
Works in bull/bear via 4h trend filter and session timing.
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
    
    # Get 4h data for trend and swing points
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 and EMA200 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 4h swing high/low (prior completed bar)
    swing_high_4h = df_4h['high'].values
    swing_low_4h = df_4h['low'].values
    # Shift by 1 to use only completed bars (prior bar's high/low)
    swing_high_4h = np.roll(swing_high_4h, 1)
    swing_low_4h = np.roll(swing_low_4h, 1)
    swing_high_4h[0] = swing_high_4h[1] if len(swing_high_4h) > 1 else swing_high_4h[0]
    swing_low_4h[0] = swing_low_4h[1] if len(swing_low_4h) > 1 else swing_low_4h[0]
    
    # Align 4h data to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    swing_high_4h_aligned = align_htf_to_ltf(prices, df_4h, swing_high_4h)
    swing_low_4h_aligned = align_htf_to_ltf(prices, df_4h, swing_low_4h)
    
    # Get daily data for Camarilla R3/S3
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    range_1d = prev_high - prev_low
    r3 = prev_close + range_1d * 1.1 / 4
    s3 = prev_close - range_1d * 1.1 / 4
    
    # Align daily data to 1h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for EMA200 and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or
            np.isnan(swing_high_4h_aligned[i]) or np.isnan(swing_low_4h_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: 4h bullish trend, price > 1d R3, breaks above prior 4h swing high
            if (ema50_4h_aligned[i] > ema200_4h_aligned[i] and  # 4h bullish
                close[i] > r3_aligned[i] and                    # above daily R3
                high[i] > swing_high_4h_aligned[i] and          # breaks 4h swing high
                vol_confirm and in_session):
                signals[i] = 0.20
                position = 1
            # Short: 4h bearish trend, price < 1d S3, breaks below prior 4h swing low
            elif (ema50_4h_aligned[i] < ema200_4h_aligned[i] and  # 4h bearish
                  close[i] < s3_aligned[i] and                    # below daily S3
                  low[i] < swing_low_4h_aligned[i] and            # breaks 4h swing low
                  vol_confirm and in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend turns bearish or price breaks below 4h swing low
            if (ema50_4h_aligned[i] < ema200_4h_aligned[i] or  # trend change
                low[i] < swing_low_4h_aligned[i]):             # break swing low
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend turns bullish or price breaks above 4h swing high
            if (ema50_4h_aligned[i] > ema200_4h_aligned[i] or  # trend change
                high[i] > swing_high_4h_aligned[i]):           # break swing high
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4H_1D_Camarilla_Structure_V1"
timeframe = "1h"
leverage = 1.0