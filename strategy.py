#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Confirmation_v4
Hypothesis: Trade breaks of Camarilla pivot levels (R1/S1) on 12h timeframe with volume confirmation. Use 1d ADX filter to avoid range-bound markets. Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels (based on prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    pivot_range = high_1d - low_1d
    r1 = close_1d + (1.1 * pivot_range / 12)
    s1 = close_1d - (1.1 * pivot_range / 12)
    
    # Align to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d ADX filter (avoid ranging markets)
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
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[0] = tr[0]
        plus_dm_smooth[0] = plus_dm[0]
        minus_dm_smooth[0] = minus_dm[0]
        
        for i in range(1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(dx)
        adx[period-1] = np.mean(dx[:period])
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Only trade when ADX > 25 (trending market)
        if adx_val < 25:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price re-enters below R1 or ADX weakens
            if price < r1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price re-enters above S1 or ADX weakens
            if price > s1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Confirmation_v4"
timeframe = "12h"
leverage = 1.0