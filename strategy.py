#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h ADX and 1d price direction for trend bias, with 1h EMA crossover for entry timing
# - Trend bias: 4h ADX > 25 (trending) AND 1d close > 1d EMA50 (uptrend) for long, < for short
# - Entry: 1h EMA12 crosses above EMA26 for long, below for short (only in trend direction)
# - Exit: Opposite EMA crossover
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Position size: 0.20 to manage risk
# - Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data for ADX once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data for EMA50 once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * np.zeros_like(high)
        minus_di = 100 * np.zeros_like(high)
        
        plus_sm = np.zeros_like(high)
        minus_sm = np.zeros_like(high)
        plus_sm[period] = np.sum(plus_dm[1:period+1])
        minus_sm[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_sm[i] = plus_sm[i-1] - (plus_sm[i-1] / period) + plus_dm[i]
            minus_sm[i] = minus_sm[i-1] - (minus_sm[i-1] / period) + minus_dm[i]
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_sm[i] + minus_sm[i] != 0:
                dx[i] = 100 * abs(plus_sm[i] - minus_sm[i]) / (plus_sm[i] + minus_sm[i])
        
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period:2*period])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate EMA50 on 1d
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h EMAs
    close_series = pd.Series(close)
    ema_12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = close_series.ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Pre-compute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    for i in range(60, n):  # Start after warmup
        # Skip if any critical data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_12[i]) or np.isnan(ema_26[i])):
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend bias conditions
        adx_trending = adx_4h_aligned[i] > 25
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long conditions: uptrend bias + EMA crossover up
            if (adx_trending and price_above_ema and 
                ema_12[i] > ema_26[i] and ema_12[i-1] <= ema_26[i-1]):
                position = 1
                signals[i] = position_size
            # Short conditions: downtrend bias + EMA crossover down
            elif (adx_trending and price_below_ema and 
                  ema_12[i] < ema_26[i] and ema_12[i-1] >= ema_26[i-1]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit long: EMA crossover down
            if ema_12[i] < ema_26[i] and ema_12[i-1] >= ema_26[i-1]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: EMA crossover up
            if ema_12[i] > ema_26[i] and ema_12[i-1] <= ema_26[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1h_4h1d_ADX_EMA_Crossover_Session"
timeframe = "1h"
leverage = 1.0