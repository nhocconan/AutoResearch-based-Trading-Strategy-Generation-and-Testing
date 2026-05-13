#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Trend_4h
Hypothesis: Camarilla R1/S1 breakouts with 4h trend filter and volume confirmation work in both bull and bear markets.
Buy when price breaks above R1 with 4h uptrend and volume spike; sell when breaks below S1 with 4h downtrend and volume spike.
Use 1d trend filter for higher timeframe bias. Target: 15-35 trades/year per symbol.
"""

name = "1h_Camarilla_R1_S1_Breakout_Trend_4h"
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
    
    # Previous day's high, low, close for Camarilla calculation
    # We'll calculate daily high/low/close from 1h data by resampling conceptually
    # But per rules, we must use get_htf_data for actual 1d data
    
    # 4h trend: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    uptrend_4h = close > ema_34_4h_aligned
    downtrend_4h = close < ema_34_4h_aligned
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate Camarilla levels using previous day's OHLC from 1d data
    # We need to align the previous day's values to each 1h bar
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's high, low, close for each 1d bar
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # Set first day's values to zero (will be handled by alignment)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla R1 and S1 for each day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    rng = prev_high - prev_low
    r1 = prev_close + 1.1 * rng / 12
    s1 = prev_close - 1.1 * rng / 12
    
    # Align to 1h timeframe (these levels are valid for the entire day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5 * 24-period average (1 day)
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after warmup for volume MA
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 4h uptrend, 1d uptrend filter, volume confirmation
            if close[i] > r1_val and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: break below S1, 4h downtrend, 1d downtrend filter, volume confirmation
            elif close[i] < s1_val and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 4h trend turns down
            if close[i] < s1_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch R1 or 4h trend turns up
            if close[i] > r1_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals