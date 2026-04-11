#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla pivot breakout with daily volume confirmation and weekly trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily chart act as strong support/resistance. 
# Breakouts above R4 or below S4 with volume confirmation indicate institutional interest.
# Weekly EMA filter ensures we only trade in direction of higher timeframe trend.
# Low frequency (~15-30/year) to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla calculation
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla: range = H - L
    # S1 = C - (range * 1.0833/2), S2 = C - (range * 1.1666/2), S3 = C - (range * 1.2500/2), S4 = C - (range * 1.5000/2)
    # R1 = C + (range * 1.0833/2), R2 = C + (range * 1.1666/2), R3 = C + (range * 1.2500/2), R4 = C + (range * 1.5000/2)
    range_1d = d_high - d_low
    s4 = d_close - (range_1d * 1.5000 / 2)
    r4 = d_close + (range_1d * 1.5000 / 2)
    
    # Align Camarilla levels to 12h timeframe
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 24-period average (2 days of 12h bars)
    vol_series = pd.Series(volume)
    vol_avg_24 = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (2.0 * vol_avg_24)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + weekly trend alignment
        if (close[i] > r4_aligned[i] and vol_confirm[i] and weekly_uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and weekly_downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to daily midpoint or trend change
        elif position == 1 and (close[i] < d_close[-1] or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > d_close[-1] or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals