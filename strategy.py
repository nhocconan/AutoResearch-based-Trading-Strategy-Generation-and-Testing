#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly ADX (14-period) for trend strength ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up = high_1w - np.roll(high_1w, 1)
    down = np.roll(low_1w, 1) - low_1w
    up[0] = 0
    down[0] = 0
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    tr14 = np.full_like(tr, np.nan)
    plus_dm14 = np.full_like(tr, np.nan)
    minus_dm14 = np.full_like(tr, np.nan)
    period = 14
    if len(tr) >= period:
        tr14[period-1] = np.sum(tr[:period])
        plus_dm14[period-1] = np.sum(plus_dm[:period])
        minus_dm14[period-1] = np.sum(minus_dm[:period])
        for i in range(period, len(tr)):
            tr14[i] = tr14[i-1] - (tr14[i-1] / period) + tr[i]
            plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / period) + plus_dm[i]
            minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / period) + minus_dm[i]
    
    # Directional Indicators
    plus_di = np.full_like(tr14, np.nan)
    minus_di = np.full_like(tr14, np.nan)
    dx = np.full_like(tr14, np.nan)
    for i in range(period-1, len(tr14)):
        if tr14[i] != 0:
            plus_di[i] = 100 * plus_dm14[i] / tr14[i]
            minus_di[i] = 100 * minus_dm14[i] / tr14[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX = smoothed DX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # === Daily Close for price action ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Align indicators to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # === Daily 20-period EMA for trend filter ===
    ema20_1d = np.full_like(close_1d, np.nan)
    alpha = 2 / (20 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema20_1d[i] = close_1d[i]
        elif np.isnan(ema20_1d[i-1]):
            ema20_1d[i] = close_1d[i]
        else:
            ema20_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema20_1d[i-1]
    
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(ema20_aligned[i]) or 
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Strong uptrend (ADX > 25) + price above EMA20
            if (adx_aligned[i] > 25 and 
                close_1d_aligned[i] > ema20_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Strong downtrend (ADX > 25) + price below EMA20
            elif (adx_aligned[i] > 25 and 
                  close_1d_aligned[i] < ema20_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Trend weakening (ADX < 20) or price crosses below EMA20
            if (adx_aligned[i] < 20 or 
                close_1d_aligned[i] < ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend weakening (ADX < 20) or price crosses above EMA20
            if (adx_aligned[i] < 20 or 
                close_1d_aligned[i] > ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_ADX25_EMA20_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0