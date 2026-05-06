#!/usr/bin/env python3
# 4h_1dCamarilla_R1S1_Breakout_1dEMA34_Trend_Volume_v4
# Uses daily Camarilla pivot levels (R1/S1) as breakout levels with daily trend filter (EMA34)
# and daily volume confirmation. Designed for 4h timeframe to capture major pivot breaks
# with trend alignment, working in both bull and bear markets by following the daily trend.
# Tightened volume threshold and added ADX filter to reduce trade frequency and improve win rate.
# Target: 75-200 total trades over 4 years (19-50/year) with 0.30 position sizing.

name = "4h_1dCamarilla_R1S1_Breakout_1dEMA34_Trend_Volume_v4"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (R1/S1 - tighter range for fewer trades)
    r1 = pp + range_1d * 1.1 / 12
    s1 = pp - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA) with higher threshold
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma_20)  # Increased threshold for stronger confirmation
    
    # ADX filter for trend strength (using 1d data)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros(len(high))
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            plus_dm_sum = np.sum(plus_dm[i-period+1:i+1])
            minus_dm_sum = np.sum(minus_dm[i-period+1:i+1])
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_sum / (atr[i] * period)
                minus_di[i] = 100 * minus_dm_sum / (atr[i] * period)
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(volume_spike[i]) or np.isnan(adx_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: break above R1 with uptrend (ADX > 20), EMA34 filter, and volume
            if close[i] > r1_4h[i] and adx_4h[i] > 20 and close[i] > ema_34_4h[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                bars_since_entry = 0
            # Short: break below S1 with downtrend (ADX > 20), EMA34 filter, and volume
            elif close[i] < s1_4h[i] and adx_4h[i] > 20 and close[i] < ema_34_4h[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit conditions: price returns to EMA34 or breaks below S1
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] < ema_34_4h[i] or close[i] < s1_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit conditions: price returns to EMA34 or breaks above R1
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] > ema_34_4h[i] or close[i] > r1_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.30
    
    return signals