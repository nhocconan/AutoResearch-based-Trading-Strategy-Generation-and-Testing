#!/usr/bin/env python3
# 6h_ADX_Trend_RSI_Momentum
# Hypothesis: Combine ADX trend strength with RSI momentum on 6h timeframe, filtered by weekly trend.
# In trending markets (ADX > 25), momentum (RSI > 50 for long, RSI < 50 for short) captures continuation.
# Weekly trend filter (price above/below weekly EMA50) ensures alignment with higher timeframe direction.
# Designed for 6h to achieve 12-37 trades/year, works in both bull and bear markets by following the trend.

name = "6h_ADX_Trend_RSI_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_rma(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        tr_smooth = smooth_rma(tr, period)
        plus_dm_smooth = smooth_rma(plus_dm, period)
        minus_dm_smooth = smooth_rma(minus_dm, period)
        
        # Directional Indicators
        plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
        minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = smooth_rma(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # RSI (14-period)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)  # First value 0
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        def smooth_rma(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        up_smooth = smooth_rma(up, period)
        down_smooth = smooth_rma(down, period)
        
        rs = np.where(down_smooth != 0, up_smooth / down_smooth, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Align weekly indicators to 6h timeframe (wait for weekly bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending), RSI > 50 (bullish momentum), price above weekly EMA50
            if adx[i] > 25 and rsi[i] > 50 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending), RSI < 50 (bearish momentum), price below weekly EMA50
            elif adx[i] > 25 and rsi[i] < 50 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (weak trend) or RSI < 40 (loss of momentum) or price below weekly EMA50
            if adx[i] < 20 or rsi[i] < 40 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (weak trend) or RSI > 60 (loss of momentum) or price above weekly EMA50
            if adx[i] < 20 or rsi[i] > 60 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals