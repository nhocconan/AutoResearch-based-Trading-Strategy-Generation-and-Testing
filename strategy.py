#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend direction (price > EMA50 = bullish trend, price < EMA50 = bearish trend).
- Camarilla levels from 1d: R3/S3 for mean reversion fade, R4/S4 for breakout continuation.
- Entry Logic:
  * In bullish 1w trend: Long on break above R4, Short on break below S3 (fade)
  * In bearish 1w trend: Short on break below S4, Long on break above R3 (fade)
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h average volume
- Signal size: 0.25 discrete to minimize fee drag
- Camarilla pivots provide mathematically derived support/resistance levels
- Weekly trend filter ensures we trade with the higher timeframe momentum
- Volume confirmation validates breakout strength
- Works in bull markets (trend continuation) and bear markets (mean reversion at extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC data.
    Returns: (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    Typical use: R3/S3 for fade, R4/S4 for breakout
    """
    typical_price = (high + low + close) / 3.0
    range_ = high - low
    
    # Calculate pivot point
    pp = typical_price
    
    # Calculate resistance and support levels
    r1 = pp + (range_ * 1.1 / 12)
    r2 = pp + (range_ * 1.1 / 6)
    r3 = pp + (range_ * 1.1 / 4)
    r4 = pp + (range_ * 1.1 / 2)
    
    s1 = pp - (range_ * 1.1 / 12)
    s2 = pp - (range_ * 1.1 / 6)
    s3 = pp - (range_ * 1.1 / 4)
    s4 = pp - (range_ * 1.1 / 2)
    
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 day for Camarilla calculation
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    # We'll shift the Camarilla levels by 1 to avoid using current day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla for each day, then shift to avoid look-ahead
    camarilla_data = []
    for i in range(len(high_1d)):
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_data.append([r4, r3, r2, r1, pp, s1, s2, s3, s4])
    
    camarilla_array = np.array(camarilla_data)
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_shifted = np.roll(camarilla_array, 1, axis=0)
    camarilla_shifted[0] = np.nan  # First value will be NaN
    
    # Extract R4, R3, S3, S4 levels
    r4_1d = camarilla_shifted[:, 0]
    r3_1d = camarilla_shifted[:, 1]
    s3_1d = camarilla_shifted[:, 7]
    s4_1d = camarilla_shifted[:, 8]
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        is_bullish_trend = curr_close > ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Exit conditions: opposite Camarilla breakout or trend change
        if position != 0:
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit long: price breaks below R3 (in bullish trend) or S3 (in bearish trend)
                if is_bullish_trend:
                    if curr_low <= r3_1d_aligned[i]:
                        exit_signal = True
                else:
                    if curr_low <= s3_1d_aligned[i]:
                        exit_signal = True
                        
            elif position == -1:  # Short position
                # Exit short: price breaks above S3 (in bearish trend) or R3 (in bullish trend)
                if is_bullish_trend:
                    if curr_high >= r3_1d_aligned[i]:
                        exit_signal = True
                else:
                    if curr_high >= s3_1d_aligned[i]:
                        exit_signal = True
            
            # Also exit on trend change
            if (position == 1 and not is_bullish_trend) or (position == -1 and is_bullish_trend):
                exit_signal = True
                
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0 and volume_confirm:
            if is_bullish_trend:
                # Bullish 1w trend: look for breakout continuation at R4 or fade at S3
                if curr_high >= r4_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif curr_low <= s3_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Bearish 1w trend: look for breakout continuation at S4 or fade at R3
                if curr_low <= s4_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                elif curr_high >= r3_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0