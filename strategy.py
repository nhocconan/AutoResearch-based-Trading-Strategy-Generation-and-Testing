#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_Breakout_1dRegime_Filter
Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation.
In trending markets (ADX>25), trade breakouts in direction of trend; in ranging markets (ADX<20), fade at extremes.
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at extremes when trend weakens.
"""

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
    
    # Get 6h data for Camarilla calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for each 6h bar (based on previous bar)
    R3_6h = np.full(len(close_6h), np.nan)
    S3_6h = np.full(len(close_6h), np.nan)
    R4_6h = np.full(len(close_6h), np.nan)
    S4_6h = np.full(len(close_6h), np.nan)
    
    for i in range(1, len(close_6h)):
        # Camarilla levels based on previous 6h bar's range
        high_prev = high_6h[i-1]
        low_prev = low_6h[i-1]
        close_prev = close_6h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R4_6h[i] = close_prev + (range_prev * 1.1 / 2)  # R4 level
            S4_6h[i] = close_prev - (range_prev * 1.1 / 2)  # S4 level
            R3_6h[i] = close_prev + (range_prev * 1.1 / 4)  # R3 level
            S3_6h[i] = close_prev - (range_prev * 1.1 / 4)  # S3 level
    
    # Align Camarilla levels to original timeframe
    R3_6h_aligned = align_htf_to_ltf(prices, df_6h, R3_6h)
    S3_6h_aligned = align_htf_to_ltf(prices, df_6h, S3_6h)
    R4_6h_aligned = align_htf_to_ltf(prices, df_6h, R4_6h)
    S4_6h_aligned = align_htf_to_ltf(prices, df_6h, S4_6h)
    
    # Get 1d data for regime filter (ADX) and trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX for regime detection
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period+1])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_6h_aligned[i]) or np.isnan(S3_6h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx = adx_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx > 25:  # Trending regime
                # Long: break above R3 with uptrend and volume spike
                long_signal = (close[i] > R3_6h_aligned[i]) and (close[i] > ema_trend) and vol_spike[i]
                # Short: break below S3 with downtrend and volume spike
                short_signal = (close[i] < S3_6h_aligned[i]) and (close[i] < ema_trend) and vol_spike[i]
            else:  # Ranging regime (ADX < 25)
                # Long: mean reversion from S3 with volume spike
                long_signal = (close[i] < S3_6h_aligned[i]) and vol_spike[i]
                # Short: mean reversion from R3 with volume spike
                short_signal = (close[i] > R3_6h_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: opposite touch or trend reversal
            exit_signal = (close[i] > R4_6h_aligned[i]) or (adx < 20 and close[i] > ema_trend)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: opposite touch or trend reversal
            exit_signal = (close[i] < S4_6h_aligned[i]) or (adx < 20 and close[i] < ema_trend)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_Pivot_Breakout_1dRegime_Filter"
timeframe = "6h"
leverage = 1.0