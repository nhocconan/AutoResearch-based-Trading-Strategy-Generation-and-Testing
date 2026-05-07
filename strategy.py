#!/usr/bin/env python3
"""
6h_RobustRange_Breakout_v1
Hypothesis: Breakouts from Bollinger Bands with volume confirmation and regime filter (ADX < 25 for ranging markets).
Trades breakouts in ranging conditions to capture mean-reversion failures, avoiding trending markets where false breakouts are common.
Uses weekly trend filter to only trade in direction of higher timeframe trend.
Target: 20-40 trades/year to minimize fee drag while maintaining edge in ranging markets.
"""

name = "6h_RobustRange_Breakout_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = ma + bb_std * bb_stddev
    lower = ma - bb_std * bb_stddev
    
    # Weekly trend filter: EMA of weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # ADX for regime filter (14-period)
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
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            plus_dm_sum = np.sum(plus_dm[:period])
            minus_dm_sum = np.sum(minus_dm[:period])
            
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = 100 * (plus_dm_sum / atr[i]) if atr[i] != 0 else 0
                minus_di[i] = 100 * (minus_dm_sum / atr[i]) if atr[i] != 0 else 0
                plus_dm_sum = plus_dm_sum - plus_dm[i-period+1] + plus_dm[i]
                minus_dm_sum = minus_dm_sum - minus_dm[i-period+1] + minus_dm[i]
        
        dx = np.zeros_like(tr)
        for i in range(len(tr)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(dx)
        if len(dx) >= period:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (ADX < 25)
        if adx[i] >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Bollinger Band with volume confirmation and weekly uptrend
            if close[i] > upper[i] and volume[i] > vol_ma[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Bollinger Band with volume confirmation and weekly downtrend
            elif close[i] < lower[i] and volume[i] > vol_ma[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to middle band or weekly trend changes
            if close[i] < ma[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle band or weekly trend changes
            if close[i] > ma[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals