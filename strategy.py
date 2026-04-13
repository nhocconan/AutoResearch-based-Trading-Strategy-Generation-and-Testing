#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h EMA(21) pullback to 4h EMA(50) in trending markets (ADX>25)
    # Long: price > 4h EMA50 AND ADX(14) > 25 AND 1h close > 1h EMA21 AND prior 1h close <= prior 1h EMA21 (pullback entry)
    # Short: price < 4h EMA50 AND ADX(14) > 25 AND 1h close < 1h EMA21 AND prior 1h close >= prior 1h EMA21 (pullback entry)
    # Exit: price crosses 4h EMA50 OR ADX < 20 (trend weakening)
    # Session filter: 08-20 UTC to avoid low-volume hours
    # Discrete position sizing (0.20) to minimize fee churn
    # Target: 15-37 trades/year (~60-150 over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA and ADX (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 4h ADX(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # +DM and -DM
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing for TR, +DM, -DM
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_14 / atr_14
    minus_di = 100 * minus_dm_14 / atr_14
    
    # DX and ADX
    dx = np.full_like(atr_14, np.nan)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = wilders_smoothing(dx, 14)
    
    # Align 4h indicators to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 1h EMA(21)
    ema_21_1h = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_21_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        # Weak trend filter: ADX < 20 (exit condition)
        weak_trend = adx_aligned[i] < 20
        
        # Pullback entry conditions
        long_pullback = (close[i] > ema_50_4h_aligned[i]) and \
                       strong_trend and \
                       (close[i] > ema_21_1h[i]) and \
                       (close[i-1] <= ema_21_1h[i-1])
        
        short_pullback = (close[i] < ema_50_4h_aligned[i]) and \
                        strong_trend and \
                        (close[i] < ema_21_1h[i]) and \
                        (close[i-1] >= ema_21_1h[i-1])
        
        # Exit conditions
        long_exit = (close[i] < ema_50_4h_aligned[i]) or weak_trend
        short_exit = (close[i] > ema_50_4h_aligned[i]) or weak_trend
        
        if long_pullback and in_session and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_pullback and in_session and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and (long_exit or not in_session):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or not in_session):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position if still in session
            if position == 1 and in_session:
                signals[i] = 0.20
            elif position == -1 and in_session:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_ema_pullback_adx_v1"
timeframe = "1h"
leverage = 1.0