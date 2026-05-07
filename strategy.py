#!/usr/bin/env python3
name = "4h_ThreeMonthHighLow_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d 3-month high and low (63 trading days approx)
    lookback = 63
    high_3m_1d = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    low_3m_1d = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align to 4h timeframe
    high_3m_1d_aligned = align_htf_to_ltf(prices, df_1d, high_3m_1d)
    low_3m_1d_aligned = align_htf_to_ltf(prices, df_1d, low_3m_1d)
    
    # 4h 20-period volume average for spike detection
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX for trend filter (ADX > 25 indicates strong trend)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(63, 34)  # Wait for 3M high/low and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(high_3m_1d_aligned[i]) or np.isnan(low_3m_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 3-month high with volume spike and strong trend
            if (close[i] > high_3m_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_4h[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short: Break below 3-month low with volume spike and strong trend
            elif (close[i] < low_3m_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_4h[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Price below 3-month low or trend weakening (ADX < 20)
            if close[i] < low_3m_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price above 3-month high or trend weakening (ADX < 20)
            if close[i] > high_3m_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 3-month high/low breaks capture major trend changes. 
# Volume spike confirms institutional participation. ADX filter ensures trades only in strong trends.
# Works in bull (breaks highs) and bear (breaks lows). Target 20-40 trades/year to minimize fee drag.