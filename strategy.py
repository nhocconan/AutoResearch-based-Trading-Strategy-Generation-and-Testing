#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d ADX regime filter
    # Long when price breaks above 20-period high + 12h volume > 1.3x 20-period average + 1d ADX > 25
    # Short when price breaks below 20-period low + 12h volume > 1.3x 20-period average + 1d ADX > 25
    # Exit when price crosses 10-period moving average in opposite direction
    # Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown
    # Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag
    # Volume filter ensures breakouts occur with institutional participation
    # ADX filter ensures we only trade in trending markets, avoiding chop
    # Weekly pivot direction from 1w timeframe filters trades: only long above weekly pivot, short below
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 12h volume average (20-period) with min_periods
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) with min_periods
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
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        for i in range(period, len(high)):
            plus_di[i] = 100 * (plus_dm[i] / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm[i] / atr[i]) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align all indicators to 6h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-calculate Donchian channels for 6h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ma_10[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (aligned)
        volume_12h_current = df_12h['volume'].values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_current)
        
        # Volume filter: current 12h volume > 1.3 * 20-period average
        volume_confirmation = vol_12h_aligned[i] > 1.3 * vol_ma_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        # Weekly pivot filter: only long above weekly pivot, short below
        above_weekly_pivot = close[i] > pivot_aligned[i]
        below_weekly_pivot = close[i] < pivot_aligned[i]
        
        # Breakout conditions with filters
        bullish_breakout = (close[i] > donchian_high[i] and 
                           volume_confirmation and 
                           trending_market and
                           above_weekly_pivot)
        bearish_breakout = (close[i] < donchian_low[i] and 
                           volume_confirmation and 
                           trending_market and
                           below_weekly_pivot)
        
        # Exit conditions: cross 10-period MA in opposite direction
        long_exit = close[i] < ma_10[i]
        short_exit = close[i] > ma_10[i]
        
        # Additional exit: price crosses weekly S1/R1 levels (mean reversion)
        long_exit_s1 = close[i] < s1_aligned[i]
        short_exit_r1 = close[i] > r1_aligned[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (long_exit or long_exit_s1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or short_exit_r1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_1w_donchian_breakout_volume_adx_pivot_v1"
timeframe = "6h"
leverage = 1.0