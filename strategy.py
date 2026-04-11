#!/usr/bin/env python3
"""
6h_1w_1d_ma_envelope_volume_v1
Strategy: 6h MA Envelope breakout with weekly trend filter and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses MA Envelope (2% deviation from 50-period MA) on 6h for breakout entries, filtered by weekly MA50 trend direction. Volume confirmation (>1.5x average volume) reduces false breakouts. Designed to work in both bull and bear markets by following the weekly trend - only taking longs in weekly uptrend and shorts in weekly downtrend. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_ma_envelope_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h MA Envelope components
    ma_len = 50
    envelope_pct = 0.02  # 2% envelope
    
    # 6h MA50
    ma_6h = pd.Series(close).ewm(span=ma_len, adjust=False, min_periods=ma_len).mean().values
    upper_env = ma_6h * (1 + envelope_pct)
    lower_env = ma_6h * (1 - envelope_pct)
    
    # Weekly MA50 for trend filter
    close_1w = df_1w['close'].values
    ma_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ma_50_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(ma_len, n):
        # Skip if any required data is invalid
        if (np.isnan(ma_6h[i]) or np.isnan(upper_env[i]) or np.isnan(lower_env[i]) or
            np.isnan(ma_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Weekly trend filter
        weekly_uptrend = price_close > ma_50_1w_aligned[i]
        weekly_downtrend = price_close < ma_50_1w_aligned[i]
        
        # MA Envelope breakout conditions
        breakout_up = price_close > upper_env[i]
        breakout_down = price_close < lower_env[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Exit conditions: price returns to MA or opposite envelope
        exit_long = position == 1 and (price_close < ma_6h[i] or price_close < lower_env[i])
        exit_short = position == -1 and (price_close > ma_6h[i] or price_close > upper_env[i])
        
        # Trading logic
        if breakout_up and vol_confirmed and weekly_uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirmed and weekly_downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals