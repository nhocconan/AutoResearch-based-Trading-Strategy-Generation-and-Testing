#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R1/S1) on 1w to define key levels,
# filtered by 1d trend (price > 1d EMA34 for long, < for short) and volume confirmation.
# Enter long when price breaks above weekly R1 with trend up and volume spike,
# enter short when price breaks below weekly S1 with trend down and volume spike.
# Exit on price reversion to weekly pivot point (PP) or trend failure.
# Designed for low frequency (7-25 trades/year) to avoid fee drift. Works in bull (catch breakouts)
# and bear (catch breakdowns) with trend filter and volume confirmation.

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels.
    Returns (PP, S1, S2, S3, R1, R2, R3)
    """
    typical = (high + low + close) / 3
    range_val = high - low
    pp = typical
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    return pp, s1, s2, s3, r1, r2, r3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    pp_1w, s1_1w, s2_1w, s3_1w, r1_1w, r2_1w, r3_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1d)  # Use same df_1w for alignment base
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Price levels
        pp = pp_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        
        if position == 0:
            # LONG: price breaks above weekly R1, trend up, volume confirmation
            if close[i] > r1 and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below weekly S1, trend down, volume confirmation
            elif close[i] < s1 and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price returns to weekly PP or trend fails
            if close[i] < pp or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to weekly PP or trend fails
            if close[i] > pp or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals