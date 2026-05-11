#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Fade at Camarilla R3/S3 levels during weak 1-week trend, but breakout continuation in strong 1-week trend.
# In strong weekly trend (price > weekly EMA50 and rising), R3/S3 breakouts signal continuation.
# In weak weekly trend (price < weekly EMA50 or falling), R3/S3 levels act as reversal points.
# Uses volume confirmation to filter false signals. Designed for 6S timeframe to capture multi-day moves.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by adapting to weekly regime.

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly close for Camarilla and trend ---
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA50 for trend strength
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_slope = ema_50_1w - np.roll(ema_50_1w, 1)
    ema_50_1w_slope[0] = 0
    ema_50_1w_slope = pd.Series(ema_50_1w_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    
    # --- Calculate Camarilla levels from previous weekly bar ---
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Use previous week's high/low/close to avoid look-ahead
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]  # fill first value
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    diff = prev_high - prev_low
    r3 = prev_close + (diff * 1.1 / 4)
    s3 = prev_close - (diff * 1.1 / 4)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    # Align weekly data to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_slope)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly EMA50 (50) and slope smoothing (3)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(ema_50_1w_slope_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend conditions
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        uptrend = ema_50_1w_slope_aligned[i] > 0
        downtrend = ema_50_1w_slope_aligned[i] < 0
        strong_uptrend = price_above_ema and uptrend
        strong_downtrend = not price_above_ema and downtrend
        weak_trend = not (strong_uptrend or strong_downtrend)
        
        if position == 0:
            if vol_surge[i]:
                # In strong uptrend: R3 breakout = continuation long
                if strong_uptrend and close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # In strong downtrend: S3 breakdown = continuation short
                elif strong_downtrend and close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # In weak trend: R3/S3 as reversal levels
                elif weak_trend:
                    # Reject at R3 in weak trend = short
                    if close[i] < r3_aligned[i] and i > start_idx and close[i-1] >= r3_aligned[i-1]:
                        signals[i] = -0.25
                        position = -1
                    # Reject at S3 in weak trend = long
                    elif close[i] > s3_aligned[i] and i > start_idx and close[i-1] <= s3_aligned[i-1]:
                        signals[i] = 0.25
                        position = 1
        else:
            if position == 1:
                # Exit long: weekly trend turns weak/down OR price crosses below S3 (support)
                if not strong_uptrend or close[i] < s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns weak/up OR price crosses above R3 (resistance)
                if not strong_downtrend or close[i] > r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals