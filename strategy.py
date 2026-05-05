#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R + 4h ADX trend filter + session filter (08-20 UTC)
# Williams %R: momentum oscillator, long when < -80 (oversold), short when > -20 (overbought)
# 4h ADX > 25 filters for trending regimes to avoid whipsaws in ranging markets
# Session filter reduces noise trades during low-liquidity hours (20-08 UTC)
# Uses 1h for entry timing, 4h for directional filter. Target: 60-150 total trades over 4 years.

name = "1h_WilliamsR_4hADX_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams %R(14) on 1h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Williams %R signals: < -80 oversold (long), > -20 overbought (short)
    oversold = williams_r < -80
    overbought = williams_r > -20
    
    # Get 4h data ONCE before loop for ADX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    if len(high_4h) >= 14:
        # True Range
        tr1 = np.abs(high_4h[1:] - low_4h[1:])
        tr2 = np.abs(high_4h[1:] - close_4h[:-1])
        tr3 = np.abs(low_4h[1:] - close_4h[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high_4h[1:] - high_4h[:-1]
        down_move = low_4h[:-1] - low_4h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
    else:
        adx = np.full(len(df_4h), np.nan)
    
    # ADX trend filter: ADX > 25 = strong trend
    adx_trend = adx > 25
    
    # Align 4h ADX to 1h timeframe
    adx_trend_aligned = align_htf_to_ltf(prices, df_4h, adx_trend.astype(float))
    
    # Session filter: 08-20 UTC (pre-compute outside loop)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(oversold[i]) or np.isnan(overbought[i]) or 
            np.isnan(adx_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold + strong trend + in session
            if (oversold[i] and 
                adx_trend_aligned[i] == 1.0 and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: overbought + strong trend + in session
            elif (overbought[i] and 
                  adx_trend_aligned[i] == 1.0 and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: overbought OR out of session
            if (overbought[i] or not in_session[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: oversold OR out of session
            if (oversold[i] or not in_session[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals