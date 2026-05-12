#!/usr/bin/env python3
# 1d_Pivots_R3S3_Breakout_WeeklyTrend_Volume
# Hypothesis: Price breaking above Camarilla R3 or below S3 on 1d with weekly trend alignment and volume confirmation captures strong moves. Weekly trend filter avoids counter-trend trades. Works in bull via breakouts above R3, in bear via breakdowns below S3. Target: 15-25 trades/year.

name = "1d_Pivots_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    range_val = high - low
    R3 = close + range_val * 1.1 / 4
    S3 = close - range_val * 1.1 / 4
    return R3, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Camarilla levels
    camarilla_R3, camarilla_S3 = calculate_camarilla(high, low, close)
    
    # Align weekly trend to daily
    weekly_trend_up = ema34_1w_aligned > 0  # placeholder, will be replaced with actual comparison
    # Actually need to compare price to EMA
    weekly_trend_up = close > ema34_1w_aligned  # price above weekly EMA34 = uptrend
    weekly_trend_down = close < ema34_1w_aligned  # price below weekly EMA34 = downtrend
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_level = camarilla_R3[i]
        s3_level = camarilla_S3[i]
        weekly_up = weekly_trend_up[i]
        weekly_down = weekly_trend_down[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with weekly uptrend and volume confirmation
            if close[i] > r3_level and weekly_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with weekly downtrend and volume confirmation
            elif close[i] < s3_level and weekly_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversal signal) or weekly trend turns down
            if close[i] < s3_level or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversal signal) or weekly trend turns up
            if close[i] > r3_level or weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals