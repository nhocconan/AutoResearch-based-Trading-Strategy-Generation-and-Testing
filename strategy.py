#!/usr/bin/env python3
"""
4h_1d_1w_Chaikin_Money_Flow_Strategy
Hypothesis: Uses Chaikin Money Flow (CMF) on daily chart to detect institutional buying/selling pressure.
Combines with weekly ADX trend filter to avoid choppy markets.
Enters long when CMF > +0.15 and price above 20-period EMA on 4h.
Enters short when CMF < -0.15 and price below 20-period EMA on 4h.
Uses weekly ADX > 20 to ensure trending market conditions.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 20-50 trades per year to minimize fee drag.
"""

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
    
    # Calculate 20-period EMA on 4h for trend filter
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get daily data for Chaikin Money Flow
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Money Flow Multiplier and Volume
    mfm = np.zeros_like(close_1d)
    mfv = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if high_1d[i] != low_1d[i]:
            mfm[i] = ((close_1d[i] - low_1d[i]) - (high_1d[i] - close_1d[i])) / (high_1d[i] - low_1d[i])
        else:
            mfm[i] = 0.0
        mfv[i] = mfm[i] * volume_1d[i]
    
    # Calculate 20-period CMF
    cmf = np.zeros_like(close_1d)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    
    for i in range(len(cmf)):
        if volume_sum[i] != 0:
            cmf[i] = mfv_sum[i] / volume_sum[i]
        else:
            cmf[i] = 0.0
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14) on weekly data using Wilder's smoothing
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        # Initialize first values
        atr[period] = np.mean(tr[1:period+1]) if len(tr) > period else 0
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1]) if len(plus_dm) > period else 0
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1]) if len(minus_dm) > period else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di = 100 * plus_dm_smooth[i] / atr[i]
                minus_di = 100 * minus_dm_smooth[i] / atr[i]
                dx[i] = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100 if (plus_di + minus_di) != 0 else 0
        
        # Calculate ADX as smoothed DX
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1]) if len(dx) > 2*period else 0
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align all signals to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(cmf_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 20 (trending market)
        strong_trend = adx_1w_aligned[i] > 20
        
        # Entry conditions: CMF extreme with EMA filter and trend
        long_entry = (cmf_aligned[i] > 0.15) and (close[i] > ema_20[i]) and strong_trend
        short_entry = (cmf_aligned[i] < -0.15) and (close[i] < ema_20[i]) and strong_trend
        
        # Exit conditions: CMF returns to neutral zone
        exit_long = position == 1 and cmf_aligned[i] < 0.05
        exit_short = position == -1 and cmf_aligned[i] > -0.05
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Chaikin_Money_Flow_Strategy"
timeframe = "4h"
leverage = 1.0