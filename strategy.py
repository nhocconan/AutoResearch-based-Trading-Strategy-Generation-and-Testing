#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h directional filter and 1d volatility filter
# Uses 4h ADX for trend strength (ADX > 25) and 1d ATR rank for volatility regime
# Enters long when price > 4h EMA(50) and ADX > 25, short when price < 4h EMA(50) and ADX > 25
# Only trades during 08-20 UTC session to avoid low-liquidity hours
# Position size 0.20 to limit drawdown
# Target: 15-30 trades/year per symbol to minimize fee drag

name = "1h_4h_adx_ema_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(df_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] - ema_4h[i-1]) * multiplier + ema_4h[i-1]
    
    # Calculate 4h ADX(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr_4h = np.zeros(len(df_4h))
    tr_4h[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(df_4h)):
        tr0 = high_4h[i] - low_4h[i]
        tr1 = abs(high_4h[i] - close_4h[i-1])
        tr2 = abs(low_4h[i] - close_4h[i-1])
        tr_4h[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_4h))
    minus_dm = np.zeros(len(df_4h))
    for i in range(1, len(df_4h)):
        up_move = high_4h[i] - high_4h[i-1]
        down_move = low_4h[i-1] - low_4h[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.sum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_4h = smooth_wilder(tr_4h, 14)
    plus_di_4h = 100 * smooth_wilder(plus_dm, 14) / atr_4h
    minus_di_4h = 100 * smooth_wilder(minus_dm, 14) / atr_4h
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h)
    adx_4h = np.full_like(dx_4h, np.nan)
    for i in range(27, len(dx_4h)):  # 14+13 for ADX smoothing
        if i == 27:
            adx_4h[i] = np.mean(dx_4h[14:28])
        else:
            adx_4h[i] = (adx_4h[i-1] * 13 + dx_4h[i]) / 14
    
    # Align 4h indicators to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # ATR percentile rank (50-day lookback)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(50, len(df_1d)):
        window = atr_1d[i-50:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align ATR rank to 1h timeframe
    atr_rank_1h = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or 
            np.isnan(atr_rank_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade in low-mid volatility (ATR rank < 70)
        if atr_rank_1h[i] >= 70:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA(50) OR ADX falls below 20
            if close[i] <= ema_4h_aligned[i] or adx_4h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA(50) OR ADX falls below 20
            if close[i] >= ema_4h_aligned[i] or adx_4h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price above EMA AND strong trend (ADX > 25)
            if close[i] > ema_4h_aligned[i] and adx_4h_aligned[i] > 25:
                position = 1
                signals[i] = 0.20
            # Enter short: price below EMA AND strong trend (ADX > 25)
            elif close[i] < ema_4h_aligned[i] and adx_4h_aligned[i] > 25:
                position = -1
                signals[i] = -0.20
    
    return signals