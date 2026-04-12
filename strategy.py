#!/usr/bin/env python3
"""
12h_1d_Camarilla_Range_Breakout
Hypothesis: On 12h timeframe, take long when price breaks above Camarilla H3 from prior day in ranging markets,
and short when price breaks below Camarilla L3. Use 1d ADX < 25 to identify ranging conditions.
Add volume confirmation > 1.5x average. Target 50-150 total trades over 4 years (12-37/year).
Works in bull markets (buying breakouts in ranges) and bear markets (selling breakdowns in ranges).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Range_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR CAMARILLA LEVELS AND ADX ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3/L3 for each day (from prior day)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        range_prev = high_1d[i-1] - low_1d[i-1]
        close_prev = close_1d[i-1]
        camarilla_h3[i] = close_prev + range_prev * 1.1 / 6
        camarilla_l3[i] = close_prev - range_prev * 1.1 / 6
    
    # Calculate ADX(14) for daily timeframe
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm[i] / atr[i]
                minus_di[i] = 100 * minus_dm[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        if len(high) >= 2*period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range condition: ADX < 25 indicates ranging market
        is_ranging = adx_12h[i] < 25
        
        # Breakout conditions
        breakout_up = close[i] > h3_12h[i]  # Close above H3
        breakout_down = close[i] < l3_12h[i]  # Close below L3
        
        # Entry conditions: breakout + ranging + volume
        long_signal = breakout_up and is_ranging and vol_ratio[i] > 1.5
        short_signal = breakout_down and is_ranging and vol_ratio[i] > 1.5
        
        # Exit conditions: return to mean (close back inside H3/L3 range) or trend emerges
        exit_long = position == 1 and close[i] < (h3_12h[i] + l3_12h[i]) / 2
        exit_short = position == -1 and close[i] > (h3_12h[i] + l3_12h[i]) / 2
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals