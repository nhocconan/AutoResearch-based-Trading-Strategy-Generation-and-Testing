#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly ADX for trend strength and daily pivot points for mean reversion entries.
# Uses 1w ADX > 25 to identify trending markets and 1d pivot points (support/resistance) for entries.
# Long when price touches 1d support (S1) in uptrend (ADX > 25 and price > 200 EMA).
# Short when price touches 1d resistance (R1) in downtrend (ADX > 25 and price < 200 EMA).
# Exit when price crosses the 1d pivot point (PP) or ADX falls below 20.
# Designed for low trade frequency (12-25/year) to avoid fee drag. Works in both trending and ranging markets.

name = "12h_1wADX_1dPivot_Pullback"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.zeros_like(close)
        plus_dm_sm = np.zeros_like(close)
        minus_dm_sm = np.zeros_like(close)
        
        # Initial values
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_sm[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_sm[period] = np.nanmean(minus_dm[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sm[i] = (plus_dm_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_sm[i] = (minus_dm_sm[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = 100 * plus_dm_sm / atr
        minus_di = 100 * minus_dm_sm / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros_like(close)
        
        # ADX smoothing
        adx[2*period] = np.nanmean(dx[period:2*period+1])
        for i in range(2*period+1, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w[:27] = np.nan  # Not enough data for ADX
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot points (standard)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    r2_1d = pp_1d + (high_1d - low_1d)
    s2_1d = pp_1d - (high_1d - low_1d)
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Align 1d pivot points to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 12h EMA 200 for trend filter
    close_s = pd.Series(close)
    ema_200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA 200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: ADX > 25 (trending), price > EMA200 (uptrend), and price at S1 support
            if (adx_1w_aligned[i] > 25 and 
                close[i] > ema_200[i] and 
                abs((close[i] - s1_1d_aligned[i]) / s1_1d_aligned[i]) < 0.005):  # Within 0.5% of S1
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (trending), price < EMA200 (downtrend), and price at R1 resistance
            elif (adx_1w_aligned[i] > 25 and 
                  close[i] < ema_200[i] and 
                  abs((close[i] - r1_1d_aligned[i]) / r1_1d_aligned[i]) < 0.005):  # Within 0.5% of R1
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above pivot point or ADX falls below 20
            if close[i] > pp_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below pivot point or ADX falls below 20
            if close[i] < pp_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals