#!/usr/bin/env python3
"""
1H_Camarilla_R3S3_Breakout_4hTrend_Volume
Hypothesis: 4h EMA50 defines trend, daily Camarilla R3/S3 levels act as strong support/resistance.
In bull markets, buy breakouts above R3 with 4h uptrend. In bear markets, sell breakdowns below S3 with 4h downtrend.
Volume spike confirms institutional interest. Uses 1h timeframe for entry timing with 4h trend filter to reduce false signals.
Target: 15-37 trades/year (60-150 over 4 years) with session filter (08-20 UTC) to avoid low-volume periods.
"""

name = "1H_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load daily data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (outer levels for fewer, stronger signals)
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.5000
    s3 = close_1d - hl_range * 1.5000
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 24-period EMA for spike detection (1h volume, ~1 day)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 2.0  # Require stronger spike for 1h
    
    # Fixed position size to minimize churn
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session or any required data is invalid
        if not in_session[i] or \
           (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema4h = close[i] > ema50_4h_aligned[i]
        price_below_ema4h = close[i] < ema50_4h_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 4h EMA50 + volume spike + session
            if breakout_long and price_above_ema4h and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 4h EMA50 + volume spike + session
            elif breakout_short and price_below_ema4h and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - trend reversal or opposite breakout
            if position == 1:
                # Exit: Price breaks below S3 OR trend reverses (close below 4h EMA)
                if close[i] < s3_aligned[i] or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks above R3 OR trend reverses (close above 4h EMA)
                if close[i] > r3_aligned[i] or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals