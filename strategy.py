#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily data provide strong intraday support/resistance. 
Breakout above R1 with daily uptrend and volume spike = long.
Breakdown below S1 with daily downtrend and volume spike = short.
Exit on opposite level touch or daily trend reversal. Uses 1w trend filter for higher timeframe bias.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Works in both bull and bear markets by following daily trend and requiring volume confirmation.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high, low, close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use the previous day's data to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    prev_daily_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_daily_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    prev_daily_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    
    # Calculate Camarilla levels
    camarilla_width = (prev_daily_high - prev_daily_low) * 1.1 / 12
    r1 = prev_daily_close + camarilla_width
    s1 = prev_daily_close - camarilla_width
    
    # Align daily levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = daily_close > ema_50_1d
    downtrend_1d = daily_close < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Weekly trend filter (for higher timeframe bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # If not enough weekly data, just use daily trend
        uptrend_1w_aligned = np.ones(n, dtype=bool)
        downtrend_1w_aligned = np.zeros(n, dtype=bool)
    else:
        weekly_close = df_1w['close'].values
        ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
        uptrend_1w = weekly_close > ema_50_1w
        downtrend_1w = weekly_close < ema_50_1w
        uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
        downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        uptrend_daily = uptrend_1d_aligned[i]
        downtrend_daily = downtrend_1d_aligned[i]
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, daily uptrend, weekly uptrend filter, volume confirmation
            if close[i] > r1_level and uptrend_daily and uptrend_weekly and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, daily downtrend, weekly downtrend filter, volume confirmation
            elif close[i] < s1_level and downtrend_daily and downtrend_weekly and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or daily trend turns down
            if close[i] < s1_level or not uptrend_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or daily trend turns up
            if close[i] > r1_level or not downtrend_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals