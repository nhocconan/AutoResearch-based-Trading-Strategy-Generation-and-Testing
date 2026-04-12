#!/usr/bin/env python3
"""
4h_1d_Keltner_Channel_Breakout_v1
Hypothesis: Use 1d Keltner Channel (20, 1.5) with 4h price breakouts and volume confirmation.
Trade long when price breaks above upper KC with volume > 1.5x average, short when breaks below lower KC.
Only trade in direction of 1d EMA50 trend to avoid counter-trend whipsaws.
Targets 20-40 trades/year to minimize fee drag. Works in bull (follow trend breakouts) and bear (fade reversals at KC bands).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Keltner_Channel_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Keltner Channel and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === DAILY KELTNER CHANNEL ===
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # EMA20 for middle line
    ema20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR10 for bands
    tr1 = np.maximum(daily_high[1:] - daily_low[1:], np.abs(daily_high[1:] - daily_close[:-1]))
    tr2 = np.maximum(tr1, np.abs(daily_low[1:] - daily_close[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # First element NaN
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Upper and lower bands (multiplier 1.5)
    upper_kc = ema20 + 1.5 * atr10
    lower_kc = ema20 - 1.5 * atr10
    
    upper_kc_4h = align_htf_to_ltf(prices, df_1d, upper_kc)
    lower_kc_4h = align_htf_to_ltf(prices, df_1d, lower_kc)
    
    # === DAILY EMA50 TREND FILTER ===
    ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50)
    
    # === VOLUME FILTER (20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(upper_kc_4h[i]) or np.isnan(lower_kc_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema50_4h[i]
        
        # Breakout conditions
        breakout_up = high[i] > upper_kc_4h[i] and vol_confirm
        breakout_down = low[i] < lower_kc_4h[i] and vol_confirm
        
        # Entry logic: only trade in direction of daily trend
        long_entry = breakout_up and trend_up
        short_entry = breakout_down and not trend_up
        
        # Exit logic: reverse signal or price returns to EMA20 (middle KC)
        ema20_4h = align_htf_to_ltf(prices, df_1d, ema20)
        long_exit = not breakout_up or close[i] < ema20_4h[i]
        short_exit = not breakout_down or close[i] > ema20_4h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals