#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_v2
Hypothesis: Use weekly Camarilla pivot levels with daily price breakouts and volume confirmation.
Trade long when price breaks above weekly H4 with volume > 1.5x average, short when breaks below weekly L4.
Only trade when daily ADX > 25 to ensure trending conditions.
Targets 15-25 trades/year to minimize fee drag. Works in bull (follow trend breakouts) and bear (avoid false breakouts in ranging markets).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Breakout_v2"
timeframe = "1d"
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
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === WEEKLY CAMARILLA PIVOT LEVELS ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Typical price
    typical_price = (weekly_high + weekly_low + weekly_close) / 3
    # Pivot point
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Ranges
    range_h = weekly_high - weekly_low
    
    # Camarilla levels
    h4 = pivot + (1.1 / 2) * range_h
    l4 = pivot - (1.1 / 2) * range_h
    h3 = pivot + (1.1 / 4) * range_h
    l3 = pivot - (1.1 / 4) * range_h
    
    h4_1d = align_htf_to_ltf(prices, df_1w, h4)
    l4_1d = align_htf_to_ltf(prices, df_1w, l4)
    h3_1d = align_htf_to_ltf(prices, df_1w, h3)
    l3_1d = align_htf_to_ltf(prices, df_1w, l3)
    
    # === DAILY ADX TREND FILTER ===
    # Calculate ADX(14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_smoothed = np.sum(plus_dm[1:period+1])
    minus_dm_smoothed = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_smoothed = plus_dm_smoothed - (plus_dm_smoothed / period) + plus_dm[i]
        minus_dm_smoothed = minus_dm_smoothed - (minus_dm_smoothed / period) + minus_dm[i]
        
        plus_di[i] = 100 * plus_dm_smoothed / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_smoothed / atr[i] if atr[i] != 0 else 0
        dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        if i == period*2:
            adx[i] = np.mean(dx[period+1:i+1])
        elif i > period*2:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
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
        if (np.isnan(h4_1d[i]) or np.isnan(l4_1d[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Breakout conditions
        breakout_up = high[i] > h4_1d[i] and vol_confirm
        breakout_down = low[i] < l4_1d[i] and vol_confirm
        
        # Entry logic: only trade breakouts in trending markets
        long_entry = breakout_up and trending
        short_entry = breakout_down and trending
        
        # Exit logic: reverse signal or price returns to H3/L3 levels
        long_exit = not breakout_up or close[i] < h3_1d[i]
        short_exit = not breakout_down or close[i] > l3_1d[i]
        
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